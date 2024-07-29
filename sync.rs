#!/usr/bin/env -S cargo +nightly -Zscript
---cargo
[dependencies]
bpaf = "0.9.12"
camino = "1.1.7"
itertools = "0.13.0"
termion = "4.0.2"
xshell = "0.2.6"

[package]
edition = "2021"

[profile.dev]
codegen-units = 16
debug = false
debug-assertions = false
incremental = false
opt-level = 3

---

use std::{
    collections::HashSet,
    ffi::OsStr,
    fmt::Display,
    fs::{self, Metadata, OpenOptions},
    io::Write as _,
    num::NonZeroUsize,
    ops::ControlFlow,
    sync::mpsc::Sender,
};

use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use itertools::Itertools as _;
use termion::{
    clear::{AfterCursor, CurrentLine},
    color::{Fg, LightGreen, LightRed},
    cursor::{DetectCursorPos as _, Goto, HideCursor, Save},
    event::Key,
    input::TermRead as _,
    raw::IntoRawMode as _,
    style::Reset,
};
use xshell::{cmd, Shell};

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T, E = BoxedError> = std::result::Result<T, E>;

const GREEN: Fg<LightGreen> = Fg(LightGreen);
const RED: Fg<LightRed> = Fg(LightRed);
const RESET: Reset = Reset;

fn main() {
    let args = parse_args();
    let Err(e) = run(args) else { return };
    eprintln!("{e}");
    std::process::exit(1);
}

fn run(args: Args) -> Result<()> {
    let verbose = args.verbose >= 1;
    let max_jobs = NonZeroUsize::new(usize::from(args.jobs))
        .or_else(|| std::thread::available_parallelism().ok())
        .map_or(1, |x| x.get());

    let sh = xshell::Shell::new()?;
    let home = sh.var("HOME")?;
    let home = Path::new(&home);

    let repo = args.repo.as_path();
    let ignorefile = repo.join(".syncignore");

    let ls_files;
    let mut files = {
        let _guard = sh.push_dir(repo);

        let ls_deleted = cmd!(sh, "git ls-files --full-name --deleted").dbg_read()?;
        let mut exclude = ls_deleted.lines().map(Path::new).collect::<HashSet<_>>();

        let ls_ignored;
        if ignorefile.exists() {
            ls_ignored = cmd!(
                sh,
                "git ls-files --full-name --exclude-from=.syncignore --ignored --cached"
            )
            .dbg_read()?;
            exclude.extend(ls_ignored.lines().map(Path::new));
        }

        ls_files = cmd!(sh, "git ls-files --full-name --cached").dbg_read()?;
        ls_files
            .lines()
            .map(Path::new)
            .filter(|file| !exclude.contains(file))
            .collect_vec()
    };

    while !files.is_empty() {
        let max_workers = files.len().min(max_jobs);
        let current_files = std::mem::take(&mut files);

        if max_workers == 1 {
            for file in current_files {
                if let Some(action) = work(
                    |action, file| {
                        act(verbose, args.dry_run, &ignorefile, &mut files, file, action)
                    },
                    repo,
                    home,
                    &sh,
                    file,
                ) {
                    match action? {
                        ControlFlow::Continue(_) => continue,
                        ControlFlow::Break(_) => break,
                    }
                }
            }
        } else {
            std::thread::scope(|s| -> Result<()> {
                let (arbiter, workers) = std::sync::mpsc::channel();
                let (send_action, recv_action) = std::sync::mpsc::channel();

                for _ in 0..max_workers {
                    s.spawn({
                        let arbiter = arbiter.clone();
                        let send_action = send_action.clone();
                        let send_action =
                            move |action, file| send_action.send((action, file)).unwrap();
                        move || worker(arbiter, send_action, repo, home)
                    });
                }

                for file in current_files {
                    let worker = workers.recv().unwrap();
                    worker.send(file).unwrap();
                }

                drop(send_action);
                drop(arbiter);
                drop(workers);

                for (action, file) in recv_action.iter() {
                    match act(verbose, args.dry_run, &ignorefile, &mut files, file, action)? {
                        ControlFlow::Continue(_) => continue,
                        ControlFlow::Break(_) => break,
                    }
                }

                Ok(())
            })?;
        }
    }

    Ok(())
}

fn act<'a>(
    verbose: bool,
    dry_run: u8,
    ignorefile: &Path,
    files: &mut Vec<&'a Path>,
    file: &'a Path,
    action: Action,
) -> Result<ControlFlow<()>> {
    let decision = handler(verbose, dry_run >= 2, action)?;
    match decision {
        Decision::Meta(meta) => match meta {
            MetaDecision::Skip => {}
            MetaDecision::Ignore => {
                if dry_run >= 1 {
                    println!("echo {file} >> {ignorefile}");
                } else {
                    let mut sync_file = OpenOptions::new().append(true).open(ignorefile)?;
                    sync_file.write_all(b"\n")?;
                    sync_file.write_all(file.as_str().as_bytes())?;
                }
            }
            MetaDecision::Retry => {
                files.push(file);
            }
            MetaDecision::Abort => {
                files.clear();
                return Ok(ControlFlow::Break(()));
            }
        },
        Decision::Copy { from, to } => {
            if dry_run >= 1 {
                println!("cp {from} {to}");
            } else {
                std::fs::copy(from, to)?;
            }
        }
        Decision::Pull { repo } => {
            if dry_run >= 1 {
                println!("cd {repo}; git pull");
            } else {
                let sh = xshell::Shell::new()?;
                sh.change_dir(repo);
                cmd!(sh, "git pull").dbg_run()?;
            }
        }
        Decision::Delete(file) => {
            if dry_run >= 1 {
                println!("rm {file}");
            } else {
                std::fs::remove_file(file)?;
            }
        }
    };

    Ok(ControlFlow::Continue(()))
}

#[derive(Clone, Debug)]
enum Decision {
    Copy { from: PathBuf, to: PathBuf },
    Pull { repo: PathBuf },
    Delete(PathBuf),
    Meta(MetaDecision),
}

#[derive(Copy, Clone, Debug)]
enum MetaDecision {
    Skip,
    Ignore,
    Retry,
    Abort,
}

fn handler(verbose: bool, really_dry_run: bool, action: Action) -> Result<Decision> {
    macro_rules! handle {
        ($home_file:expr, $repo_file:expr, $action:expr) => {
            return Ok(match $action? {
                Handled::CopyFromHomeToRepo => Decision::Copy {
                    from: $home_file,
                    to: $repo_file,
                },
                Handled::CopyFromRepoToHome => Decision::Copy {
                    from: $repo_file,
                    to: $home_file,
                },
                Handled::PullHome => Decision::Pull { repo: $home_file },
                Handled::PullRepo => Decision::Pull { repo: $repo_file },
                Handled::DeleteFromRepo => Decision::Delete($repo_file),
                Handled::Meta(meta) => Decision::Meta(meta),
            });
        };
    }

    match action {
        Action::FileDiff {
            home_file,
            repo_file,
            diff,
        } => {
            handle!(
                home_file,
                repo_file,
                handle_file_diff(verbose, really_dry_run, &home_file, &repo_file, &diff)
            );
        }
        Action::DifferentRemotes {
            home_file,
            home_remote,
            repo_file,
            repo_remote,
        } => {
            handle!(
                home_file,
                repo_file,
                handle_different_remotes(
                    really_dry_run,
                    &home_file,
                    &home_remote,
                    &repo_file,
                    &repo_remote,
                )
            );
        }
        Action::DifferentHashes {
            home_file,
            home_hash,
            home_show,
            repo_file,
            repo_hash,
            repo_show,
        } => {
            handle!(
                home_file,
                repo_file,
                handle_different_hashes(
                    verbose,
                    really_dry_run,
                    (&home_file, &home_hash, &home_show),
                    (&repo_file, &repo_hash, &repo_show),
                )
            );
        }
        Action::LinkDiff {
            home_file,
            home_link,
            repo_file,
            repo_link,
            repo_at_home_link,
        } => {
            handle!(
                home_file,
                repo_file,
                handle_link_diff(
                    really_dry_run,
                    &home_file,
                    &home_link,
                    &repo_file,
                    &repo_link,
                    &repo_at_home_link,
                )
            );
        }
        Action::FileMissing {
            repo_file,
            repo_class,
            would_be_home_file,
        } => {
            if matches!(repo_class, ValidFileType::File(_)) {
                handle!(
                    would_be_home_file,
                    repo_file,
                    handle_missing_file(really_dry_run, &would_be_home_file, &repo_file)
                );
            } else if verbose {
                println!("TODO: file is missing: repo={repo_file} type={repo_class:?}");
            }
        }
        Action::NotAFile {
            repo_file,
            home_file,
            home_class,
        } => {
            if verbose {
                println!(
                    "TODO: file is not a file: repo={repo_file} home={home_file} type={home_class:?}"
                );
            }
        }
        Action::NotADir {
            repo_file,
            repo_dir,
            home_file,
            home_class,
        } => {
            if verbose {
                println!(
                    "TODO: file is not a dir: repo={repo_file} home={home_file} type={home_class:?} repo_type={repo_dir:?}"
                );
            }
        }
        Action::NotALink {
            repo_file,
            repo_link,
            home_file,
            home_class,
        } => {
            if verbose {
                println!(
                    "TODO: file is not a link: repo={repo_file} home={home_file} type={home_class:?} repo_target={repo_link}"
                );
            }
        }
    };

    Ok(Decision::Meta(MetaDecision::Skip))
}

#[derive(Copy, Clone, Debug)]
enum Handled {
    CopyFromHomeToRepo,
    CopyFromRepoToHome,
    DeleteFromRepo,
    PullHome,
    PullRepo,
    Meta(MetaDecision),
}

fn handle_file_diff(
    verbose: bool,
    really_dry_run: bool,
    home_file: &Path,
    repo_file: &Path,
    diff: &str,
) -> Result<Handled> {
    println!("The files differ between the repo and home");
    println!("Home file: {GREEN}{home_file}{RESET}");
    println!("Repo file: {GREEN}{repo_file}{RESET}");

    if !really_dry_run || verbose {
        println!("The diff is (home on the left, repo on the right):\n");
        println!("{diff}\n");
    }

    make_decision(
        [
            Options::Handled(Handled::CopyFromRepoToHome),
            Options::Handled(Handled::CopyFromHomeToRepo),
            Options::Handled(Handled::Meta(MetaDecision::Skip)),
            Options::EditHomeAndRetry,
            Options::EditRepoAndRetry,
            Options::ShellAndRetry,
            Options::Handled(Handled::Meta(MetaDecision::Ignore)),
            Options::Handled(Handled::Meta(MetaDecision::Abort)),
        ],
        really_dry_run,
        home_file,
        repo_file,
    )
}

fn handle_different_remotes(
    really_dry_run: bool,
    home_file: &Path,
    home_remote: &str,
    repo_file: &Path,
    repo_remote: &str,
) -> Result<Handled> {
    println!("The remote for the directories are different");
    println!("Home: {home_file} => {GREEN}{home_remote}{RESET}");
    println!("Repo: {repo_file} => {GREEN}{repo_remote}{RESET}");

    make_decision(
        [
            Options::Handled(Handled::Meta(MetaDecision::Skip)),
            Options::ShellAndRetry,
            Options::Handled(Handled::Meta(MetaDecision::Ignore)),
            Options::Handled(Handled::Meta(MetaDecision::Abort)),
        ],
        really_dry_run,
        home_file,
        repo_file,
    )
}

fn handle_different_hashes(
    verbose: bool,
    really_dry_run: bool,
    (home_file, home_hash, home_show): (&Path, &str, &str),
    (repo_file, repo_hash, repo_show): (&Path, &str, &str),
) -> Result<Handled> {
    println!("The hashes for the directories are different");
    println!("Home: {home_file} => {GREEN}{home_hash}{RESET}");
    println!("Repo: {repo_file} => {GREEN}{repo_hash}{RESET}");

    if !really_dry_run || verbose {
        println!("The commit info (home first, repo second):\n");
        println!("{home_show}\n");
        println!("{repo_show}\n");
    }

    make_decision(
        [
            Options::Handled(Handled::PullHome),
            Options::Handled(Handled::PullRepo),
            Options::Handled(Handled::Meta(MetaDecision::Skip)),
            Options::ShellAndRetry,
            Options::Handled(Handled::Meta(MetaDecision::Ignore)),
            Options::Handled(Handled::Meta(MetaDecision::Abort)),
        ],
        really_dry_run,
        home_file,
        repo_file,
    )
}

fn handle_link_diff(
    really_dry_run: bool,
    home_file: &Path,
    home_target: &Path,
    repo_file: &Path,
    repo_target: &Path,
    repo_at_home_target: &Path,
) -> Result<Handled> {
    println!("The links will point to different targets");
    println!("Home target: {home_file} => {GREEN}{home_target}{RESET}");
    println!("Repo target: {repo_file} => {GREEN}{repo_target}{RESET}");
    println!("Repo target (if it was copied to home): {GREEN}{repo_at_home_target}{RESET}");

    make_decision(
        [
            Options::Handled(Handled::CopyFromRepoToHome),
            Options::Handled(Handled::CopyFromHomeToRepo),
            Options::Handled(Handled::Meta(MetaDecision::Skip)),
            Options::ShellAndRetry,
            Options::Handled(Handled::Meta(MetaDecision::Ignore)),
            Options::Handled(Handled::Meta(MetaDecision::Abort)),
        ],
        really_dry_run,
        home_file,
        repo_file,
    )
}

fn handle_missing_file(
    really_dry_run: bool,
    home_file: &Path,
    repo_file: &Path,
) -> Result<Handled> {
    println!("The file is missing at the home location");
    println!("Repo: {GREEN}{repo_file}{RESET}");
    println!("Missing file: {RED}{home_file}{RESET}");

    make_decision(
        [
            Options::Handled(Handled::CopyFromRepoToHome),
            Options::Handled(Handled::DeleteFromRepo),
            Options::Handled(Handled::Meta(MetaDecision::Skip)),
            Options::ShellAndRetry,
            Options::Handled(Handled::Meta(MetaDecision::Ignore)),
            Options::Handled(Handled::Meta(MetaDecision::Abort)),
        ],
        really_dry_run,
        home_file,
        repo_file,
    )
}

fn make_decision<const N: usize>(
    options: [Options; N],
    really_dry_run: bool,
    home_file: impl AsRef<OsStr>,
    repo_file: impl AsRef<OsStr>,
) -> Result<Handled> {
    let decision = if really_dry_run {
        Options::Handled(Handled::Meta(MetaDecision::Skip))
    } else {
        let decision = selection_prompt(options, "What do you want to do?")?;
        let decision = decision.unwrap_or(Options::Handled(Handled::Meta(MetaDecision::Abort)));

        println!("{decision}\n");

        decision
    };

    apply_decision(decision, home_file, repo_file)
}

fn apply_decision(
    decision: Options,
    home_file: impl AsRef<OsStr>,
    repo_file: impl AsRef<OsStr>,
) -> Result<Handled> {
    Ok(match decision {
        Options::Handled(handled) => handled,
        Options::EditHomeAndRetry => {
            if std::process::Command::new(std::env::var_os("EDITOR").unwrap_or_else(|| "vi".into()))
                .arg(home_file)
                .status()?
                .success()
            {
                Handled::Meta(MetaDecision::Retry)
            } else {
                Handled::Meta(MetaDecision::Skip)
            }
        }
        Options::EditRepoAndRetry => {
            if std::process::Command::new(std::env::var_os("EDITOR").unwrap_or_else(|| "vi".into()))
                .arg(repo_file)
                .status()?
                .success()
            {
                Handled::Meta(MetaDecision::Retry)
            } else {
                Handled::Meta(MetaDecision::Skip)
            }
        }
        Options::ShellAndRetry => {
            if std::process::Command::new(std::env::var_os("SHELL").unwrap_or_else(|| "sh".into()))
                .envs([
                    ("HOME_FILE", home_file.as_ref()),
                    ("REPO_FILE", repo_file.as_ref()),
                ])
                .status()?
                .success()
            {
                Handled::Meta(MetaDecision::Retry)
            } else {
                Handled::Meta(MetaDecision::Skip)
            }
        }
    })
}

#[derive(Copy, Clone, Debug)]
enum Options {
    Handled(Handled),
    EditHomeAndRetry,
    EditRepoAndRetry,
    ShellAndRetry,
}

impl Display for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Handled(handled) => match handled {
                Handled::CopyFromRepoToHome => write!(f, "Copy the file from the repo to home"),
                Handled::CopyFromHomeToRepo => write!(f, "Copy the file from home to the repo"),
                Handled::DeleteFromRepo => write!(f, "Delete the file from the repo"),
                Handled::PullHome => write!(f, "run `git pull` in home"),
                Handled::PullRepo => write!(f, "run `git pull` in the repo"),
                Handled::Meta(meta) => match meta {
                    MetaDecision::Skip => write!(f, "Skip this file"),
                    MetaDecision::Ignore => write!(f, "Ignore this file now and in the future"),
                    MetaDecision::Abort => write!(f, "Skip all remaining files and abort"),
                    MetaDecision::Retry => write!(f, "Just retry"),
                },
            },
            Self::EditHomeAndRetry => write!(f, "Edit the file in home and retry"),
            Self::EditRepoAndRetry => write!(f, "Edit the file in the repo and retry"),
            Self::ShellAndRetry => write!(f, "Drop to the shell and retry"),
        }
    }
}

impl SelectOption for Options {
    fn text(&self) -> &impl Display {
        self
    }

    fn main_key(&self) -> char {
        match self {
            Options::Handled(handled) => match handled {
                Handled::CopyFromRepoToHome => 'c',
                Handled::CopyFromHomeToRepo => 'C',
                Handled::DeleteFromRepo => 'D',
                Handled::PullHome => 'p',
                Handled::PullRepo => 'P',
                Handled::Meta(meta) => match meta {
                    MetaDecision::Skip => 's',
                    MetaDecision::Ignore => 'I',
                    MetaDecision::Abort => 'Q',
                    MetaDecision::Retry => '@',
                },
            },
            Options::EditHomeAndRetry => 'e',
            Options::EditRepoAndRetry => 'E',
            Options::ShellAndRetry => '$',
        }
    }

    fn also_matches(&self, &key: &Key) -> bool {
        matches!(
            (self, key),
            (
                Self::Handled(Handled::Meta(MetaDecision::Abort)),
                Key::Char('q')
            ) | (
                Self::Handled(Handled::Meta(MetaDecision::Ignore)),
                Key::Char('i')
            ) | (Self::Handled(Handled::Meta(MetaDecision::Skip)), Key::Esc),
        )
    }
}

fn selection_prompt<'a, T: SelectOption, P: Into<Option<&'a str>>>(
    options: impl IntoIterator<Item = T>,
    prompt: P,
) -> Result<Option<T>> {
    let prompt = prompt.into().unwrap_or("What do you want to do?");

    let tty = termion::get_tty()?;
    let mut tty = tty.into_raw_mode()?;
    let mut tty = HideCursor::from(tty.by_ref());

    write!(tty, "{}", Save)?;
    let (x, y) = tty.cursor_pos()?;

    writeln!(tty, "{}{}", CurrentLine, prompt)?;

    let mut opts = OptItem::from(options).collect_vec();
    for opt in &opts {
        writeln!(tty, "{}{}{}", Goto(x, y + opt.index), CurrentLine, opt)?;
    }
    tty.flush()?;

    let decision = tty
        .try_clone()?
        .keys()
        .filter_map(|key| key.ok())
        .find_map(|key| {
            if let Some(opt) = opts.iter().find(|opt| opt.opt.matches(&key)) {
                return Some(Some(opt.index as usize));
            }

            match key {
                Key::Ctrl('c' | 'd' | 'z') => Some(None),
                Key::Char(k) if k.is_ascii_digit() => k
                    .to_digit(10)
                    .map(|k| k as usize)
                    .filter(|&k| k <= opts.len())
                    .map(Some),
                _ => None,
            }
        })
        .and_then(|key| key.map(|k| k - 1));

    let items = opts.len() as u16;
    write!(tty, "{}{}", Goto(x, y - items + 1), AfterCursor)?;
    tty.flush()?;

    drop(tty);

    let Some(decision) = decision else {
        return Ok(None);
    };
    let decision = opts.swap_remove(decision);
    Ok(Some(decision.opt))
}

trait SelectOption {
    fn text(&self) -> &impl Display;

    fn main_key(&self) -> char;

    fn also_matches(&self, key: &Key) -> bool;

    fn matches(&self, key: &Key) -> bool {
        self.also_matches(key) || *key == Key::Char(self.main_key())
    }
}

#[derive(Copy, Clone, Debug)]
struct OptItem<T> {
    index: u16,
    opt: T,
}

impl<T> OptItem<T> {
    fn new(index: u16, opt: T) -> Self {
        Self { index, opt }
    }

    fn from(opts: impl IntoIterator<Item = T>) -> impl Iterator<Item = Self> {
        opts.into_iter()
            .enumerate()
            .map(|(index, item)| Self::new(index as u16 + 1, item))
    }
}

impl<T: SelectOption> Display for OptItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}) {}{}{} {}",
            self.index,
            termion::color::Fg(termion::color::Blue),
            self.opt.main_key(),
            termion::style::Reset,
            self.opt.text()
        )
    }
}

fn worker<'a>(
    arbiter: Sender<Sender<&'a Path>>,
    handler: impl Fn(Action, &'a Path),
    repo: &'a Path,
    home: &'a Path,
) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let sh = xshell::Shell::new().unwrap();

    loop {
        if arbiter.send(sender.clone()).is_err() {
            break;
        }

        let Ok(file) = receiver.recv() else {
            break;
        };

        work(&handler, repo, home, &sh, file);
    }
}

fn work<'a, R>(
    mut handler: impl FnMut(Action, &'a Path) -> R,
    repo: &'a Path,
    home: &'a Path,
    sh: &Shell,
    file: &'a Path,
) -> Option<R> {
    let repo_file = repo.join(file);
    let guard = sh.push_dir(repo);
    let repo_class = FileType::classify(repo, file, sh);
    drop(guard);

    let FileType::Valid(repo_class) = repo_class else {
        unreachable!("lol what: {file}");
    };

    let home_file = home.join(file);
    let guard = sh.push_dir(home);
    let home_class = FileType::classify(home, file, sh);
    drop(guard);

    let mut handler = |action| handler(action, file);

    let FileType::Valid(home_class) = home_class else {
        return Some(handler(Action::FileMissing {
            repo_file,
            repo_class,
            would_be_home_file: home_file,
        }));
    };

    use ValidFileType::*;

    match (repo_class, home_class) {
        (File(rm), File(hm)) => {
            let diff = FileDiff::diff((&home_file, &hm), (&repo_file, &rm), sh);
            if let FileDiff::Different(diff) = diff {
                return Some(handler(Action::FileDiff {
                    home_file,
                    repo_file,
                    diff,
                }));
            }
        }
        (Link(rl), Link(hl)) => {
            let would_be_home_target = if rl.is_relative() {
                home.join(&rl)
            } else {
                rl.to_path_buf()
            };
            let home_target = if hl.is_relative() {
                &home.join(&hl)
            } else {
                &hl
            };
            if would_be_home_target.as_path() != home_target {
                return Some(handler(Action::LinkDiff {
                    home_file,
                    home_link: hl,
                    repo_file,
                    repo_link: rl,
                    repo_at_home_link: would_be_home_target,
                }));
            }
        }
        (Dir(rd), Dir(hd)) => {
            if let (DirKind::Submodule, DirKind::GitDir) = (rd.kind, hd.kind) {
                if rd.remote != hd.remote {
                    return Some(handler(Action::DifferentRemotes {
                        home_file,
                        home_remote: hd.remote,
                        repo_file,
                        repo_remote: rd.remote,
                    }));
                }
                if rd.hash != hd.hash {
                    let rh = rd.hash.as_str();
                    let hh = hd.hash.as_str();
                    let repo_show = {
                        let _guard = sh.push_dir(&repo_file);
                        cmd!(sh, "git log -n1 {rh}")
                            .quiet()
                            .dbg_read()
                            .unwrap_or_default()
                    };

                    let home_show = {
                        let _guard = sh.push_dir(&home_file);
                        cmd!(sh, "git log -n1 {hh}")
                            .quiet()
                            .dbg_read()
                            .unwrap_or_default()
                    };

                    return Some(handler(Action::DifferentHashes {
                        home_file,
                        home_hash: hd.hash,
                        home_show,
                        repo_file,
                        repo_hash: rd.hash,
                        repo_show,
                    }));
                }
            };
        }
        (File(_), home_class @ (Link(_) | Dir(_))) => {
            return Some(handler(Action::NotAFile {
                repo_file,
                home_file,
                home_class,
            }));
        }
        (Dir(rd), home_class @ (Link(_) | File(_))) => {
            return Some(handler(Action::NotADir {
                repo_file,
                repo_dir: rd,
                home_file,
                home_class,
            }));
        }
        (Link(rl), home_class @ (Dir(_) | File(_))) => {
            return Some(handler(Action::NotALink {
                repo_file,
                repo_link: rl,
                home_file,
                home_class,
            }));
        }
    };

    None
}

enum Action {
    FileDiff {
        home_file: PathBuf,
        repo_file: PathBuf,
        diff: String,
    },
    LinkDiff {
        home_file: PathBuf,
        home_link: PathBuf,
        repo_file: PathBuf,
        repo_link: PathBuf,
        repo_at_home_link: PathBuf,
    },
    DifferentRemotes {
        home_file: PathBuf,
        home_remote: String,
        repo_file: PathBuf,
        repo_remote: String,
    },
    DifferentHashes {
        home_file: PathBuf,
        home_hash: String,
        home_show: String,
        repo_file: PathBuf,
        repo_hash: String,
        repo_show: String,
    },
    FileMissing {
        repo_file: PathBuf,
        repo_class: ValidFileType,
        would_be_home_file: PathBuf,
    },
    NotAFile {
        repo_file: PathBuf,
        home_file: PathBuf,
        home_class: ValidFileType,
    },
    NotADir {
        repo_file: PathBuf,
        repo_dir: DirType,
        home_file: PathBuf,
        home_class: ValidFileType,
    },
    NotALink {
        repo_file: PathBuf,
        repo_link: PathBuf,
        home_file: PathBuf,
        home_class: ValidFileType,
    },
}

#[derive(Debug, Clone)]
enum FileType {
    Missing,
    Invalid,
    Valid(ValidFileType),
}

#[derive(Debug, Clone)]
enum ValidFileType {
    Dir(DirType),
    Link(PathBuf),
    File(Metadata),
}

impl FileType {
    fn classify(root: &Path, relative: &Path, sh: &Shell) -> Self {
        let path = root.join(relative);

        let Ok(meta) = fs::symlink_metadata(&path) else {
            return Self::Missing;
        };

        let ft = meta.file_type();

        if ft.is_symlink() {
            let target = fs::read_link(&path).expect("file exists and is a symlink");
            let target = PathBuf::from_path_buf(target).expect("symlink target is not utf8");
            return Self::Valid(ValidFileType::Link(target));
        }

        if let Some(dir) = DirType::classify(relative, &meta, sh) {
            return Self::Valid(ValidFileType::Dir(dir));
        }

        if ft.is_file() {
            return Self::Valid(ValidFileType::File(meta));
        }

        Self::Invalid
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirType {
    remote: String,
    hash: String,
    kind: DirKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum DirKind {
    Submodule,
    GitDir,
}

impl DirType {
    fn classify(dir: &Path, meta: &Metadata, sh: &Shell) -> Option<Self> {
        if !meta.is_dir() {
            return None;
        }

        if let Some(submodule) = Self::try_submodule(dir, sh) {
            return Some(submodule);
        }

        if let Some(git_dir) = Self::try_git_dir(dir, sh) {
            return Some(git_dir);
        }

        None
    }

    fn try_submodule(dir: &Path, sh: &Shell) -> Option<Self> {
        let submodule_hash = cmd!(sh, "git submodule status -- {dir}")
            .ignore_err()
            .dbg_read()
            .ok()?;
        let submodule_hash = submodule_hash.get(1..)?.split_once(' ')?.0;

        let _guard = sh.push_dir(dir);
        let remote = cmd!(sh, "git ls-remote --get-url").dbg_read().ok()?;
        let remote = Self::normalize_remote(&remote);

        Some(Self {
            remote,
            hash: submodule_hash.to_owned(),
            kind: DirKind::Submodule,
        })
    }

    fn try_git_dir(dir: &Path, sh: &Shell) -> Option<Self> {
        let _guard = sh.push_dir(dir);
        let remote = cmd!(sh, "git ls-remote --get-url")
            .ignore_err()
            .dbg_read()
            .ok()?;

        if remote.trim().is_empty() {
            return None;
        }

        let hash = cmd!(sh, "git rev-parse HEAD").dbg_read().ok()?;
        let remote = Self::normalize_remote(&remote);

        Some(Self {
            remote,
            hash,
            kind: DirKind::GitDir,
        })
    }

    fn normalize_remote(remote: &str) -> String {
        remote
            .trim_start_matches("https://github.com/")
            .trim_start_matches("git@github.com:")
            .trim_end_matches(".git")
            .to_owned()
    }
}

#[derive(Debug, Clone)]
enum FileDiff {
    Different(String),
    Same,
}

impl FileDiff {
    fn diff(
        (home_file, home_meta): (&Path, &Metadata),
        (repo_file, repo_meta): (&Path, &Metadata),
        sh: &Shell,
    ) -> Self {
        if repo_meta.len() != home_meta.len() {
            return Self::Different(FileDiff::diff_content(home_file, repo_file, sh));
        }

        let hashes = cmd!(sh, "git hash-object -t blob {home_file} {repo_file}")
            .quiet()
            .dbg_read()
            .expect("git hash-object to not fail on two existing files");
        let (home_hash, repo_hash) = hashes
            .lines()
            .collect_tuple()
            .expect("git hash-object to return two hashes");

        if repo_hash != home_hash {
            return Self::Different(FileDiff::diff_content(home_file, repo_file, sh));
        }

        let is_different = cmd!(sh, "difft --check-only {home_file} {repo_file}")
            .quiet()
            .ignore_stdout()
            .dbg_run()
            .is_err();

        if is_different {
            Self::Different(FileDiff::diff_content(home_file, repo_file, sh))
        } else {
            Self::Same
        }
    }

    fn diff_content(home_file: &Path, repo_file: &Path, sh: &Shell) -> String {
        let diff_content = cmd!(
            sh,
            "difft --color=always --skip-unchanged {home_file} {repo_file}"
        )
        .quiet()
        .dbg_read();
        diff_content.unwrap_or_default()
    }
}

trait CmdExt {
    type Err;

    fn ignore_err(self) -> Self;

    fn dbg_read(self) -> Result<String, Self::Err>;

    fn dbg_run(self) -> Result<(), Self::Err>;
}

impl CmdExt for xshell::Cmd<'_> {
    type Err = xshell::Error;

    fn ignore_err(self) -> Self {
        self.quiet().ignore_stderr()
    }

    fn dbg_read(self) -> Result<String, Self::Err> {
        let dbg = print_dbg(&self);

        let output = self.read();

        if let Some(dbg) = dbg {
            eprintln!("{dbg}\n\t=>{output:?}");
        }

        output
    }

    fn dbg_run(self) -> Result<(), Self::Err> {
        let dbg = print_dbg(&self);

        let output = self.run();

        if let Some(dbg) = dbg {
            eprintln!("{dbg}\n\t=>{output:?}");
        }

        output
    }
}

fn print_dbg(command: &xshell::Cmd<'_>) -> Option<String> {
    if cfg!(debug_assertions) {
        let cmd = format!("{command:#?}");
        let cwd = cmd
            .lines()
            .skip_while(|l| !l.contains("cwd"))
            .nth(1)
            .and_then(|l| Some(l.split_once("value: ")?.1))
            .map(|s| s.trim_end_matches(',').trim_matches('"'))
            .unwrap_or_default();
        Some(format!("Running command in {cwd}: `{command}`"))
    } else {
        None
    }
}

#[derive(Clone, Debug)]
struct Args {
    dry_run: u8,
    verbose: u8,
    jobs: u8,
    repo: PathBuf,
}

fn parse_args() -> Args {
    use bpaf::{construct, positional, short, Parser};

    let dry_run = short('n')
        .long("dry-run")
        .help("Do not perform any action")
        .req_flag(())
        .count()
        .map(|v| v.clamp(0, 2) as u8);

    let verbose = short('v')
        .long("verbose")
        .help("Print more information")
        .req_flag(())
        .count()
        .map(|v| v.clamp(0, 2) as u8);

    let jobs = short('j')
        .long("jobs")
        .help("The number of threads to use. 0 selects based on CPU cores. 1 disables multithreading.")
        .argument::<u8>("JOBS")
        .fallback(0)
        .display_fallback();

    let repo = positional::<PathBuf>("REPO")
        .help("The path to the repo to sync")
        .fallback(PathBuf::from("."))
        .debug_fallback();

    let args = construct!(Args {
        dry_run,
        verbose,
        jobs,
        repo
    });

    args.to_options()
        .version(env!("CARGO_PKG_VERSION"))
        .descr("Synchronize dotfiles between home and the repo")
        .run()
}
