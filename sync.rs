#!/usr/bin/env -S cargo +nightly --quiet -Zscript
---cargo
[dependencies]
bpaf = "0.9.12"
camino = "1.1.7"
termion = "4.0.2"
xshell = "0.2.6"

[dependencies.tiny-select]
path = "/Users/knut/dev/Rust/tiny-select"

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
    collections::{HashSet, VecDeque},
    env,
    ffi::OsString,
    fmt::Display,
    fs::{self, File, Metadata, OpenOptions},
    hash::{DefaultHasher, Hasher},
    io::{self, Read as _, Write as _},
    num::NonZeroUsize,
    ops::ControlFlow,
    process::Command,
    sync::{mpsc::Sender, LazyLock},
};

use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
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
    let Err(e) = real_main() else { return };
    eprintln!("{e}");
    std::process::exit(1);
}

fn real_main() -> Result<()> {
    let sh = xshell::Shell::new()?;
    let home = sh.var("HOME")?;
    let home = PathBuf::from(home);
    let args = parse_args(home);

    run(args, &sh)
}

fn run(args: Args, sh: &Shell) -> Result<()> {
    let verbose = args.verbose >= 1;
    let max_jobs = NonZeroUsize::new(usize::from(args.jobs))
        .or_else(|| std::thread::available_parallelism().ok())
        .map_or(1, |x| x.get());

    let home = args.home.as_path();
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
            .collect::<VecDeque<_>>()
    };

    let max_workers = files.len().min(max_jobs);

    if max_workers == 1 {
        while let Some(file) = files.pop_front() {
            if let Some(action) = work(
                |action, file| act(verbose, args.dry_run, &ignorefile, &mut files, file, action),
                repo,
                home,
                sh,
                file,
            ) {
                match action? {
                    ControlFlow::Continue(_) => continue,
                    ControlFlow::Break(_) => break,
                }
            }
        }
    } else {
        while !files.is_empty() {
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

                for file in files.drain(..) {
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
    files: &mut VecDeque<&'a Path>,
    file: &'a Path,
    action: Action,
) -> Result<ControlFlow<()>> {
    let decision = handler(verbose, dry_run >= 2, action)?;
    let decision =
        try_act(dry_run, ignorefile, file, decision).or_else(|e| recover_act(file, e))?;
    Ok(match decision {
        Acted::Continue => ControlFlow::Continue(()),
        Acted::Retry(file) => {
            files.push_front(file);
            ControlFlow::Continue(())
        }
        Acted::Stop => {
            files.clear();
            ControlFlow::Break(())
        }
    })
}

enum Acted<'a> {
    Continue,
    Retry(&'a Path),
    Stop,
}

fn try_act<'a>(
    dry_run: u8,
    ignorefile: &Path,
    file: &'a Path,
    decision: Decision,
) -> Result<Acted<'a>> {
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
                return Ok(Acted::Retry(file));
            }
            MetaDecision::Abort => {
                return Ok(Acted::Stop);
            }
        },
        Decision::Copy(Copy { from, to, .. }) => {
            if dry_run >= 1 {
                println!("cp {from} {to}");
            } else {
                to.parent().map(fs::create_dir_all).transpose()?;
                fs::copy(from, to)?;
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

    Ok(Acted::Continue)
}

fn recover_act<E: Display>(file: &Path, err: E) -> Result<Acted<'_>> {
    println!("Last action failed: {err}");
    println!();

    let selected = selection_prompt(
        [
            ('R', "Retry file"),
            ('I', "Ignore"),
            ('$', "Drop to the shell and then retry the file"),
            ('Q', "Quit"),
        ],
        "What now?",
    )?;

    Ok(match selected {
        Some(('I', _)) => Acted::Continue,
        Some(('R', _)) => Acted::Retry(file),
        Some(('$', _)) => {
            run_shell([])?;
            Acted::Retry(file)
        }
        _ => Acted::Stop,
    })
}

#[derive(Clone, Debug)]
struct Copy {
    from: PathBuf,
    to: PathBuf,
}

#[derive(Clone, Debug)]
enum Decision {
    Copy(Copy),
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
        ($action:expr) => {
            return Ok(match $action? {
                Handled::Copy(cp) => Decision::Copy(cp),
                Handled::Delete(f) => Decision::Delete(f),
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
            handle!(handle_file_diff(
                verbose,
                really_dry_run,
                &home_file,
                &repo_file,
                &diff
            ));
        }
        Action::DifferentRemotes {
            home_file,
            home_remote,
            repo_file,
            repo_remote,
        } => {
            handle!(handle_different_remotes(
                really_dry_run,
                &home_file,
                &home_remote,
                &repo_file,
                &repo_remote,
            ));
        }
        Action::DifferentHashes {
            home_file,
            home_hash,
            home_show,
            repo_file,
            repo_hash,
            repo_show,
        } => {
            handle!(handle_different_hashes(
                verbose,
                really_dry_run,
                (&home_file, &home_hash, &home_show),
                (&repo_file, &repo_hash, &repo_show),
            ));
        }
        Action::LinkDiff {
            home_file,
            home_link,
            repo_file,
            repo_link,
            repo_at_home_link,
        } => {
            handle!(handle_link_diff(
                really_dry_run,
                &home_file,
                &home_link,
                &repo_file,
                &repo_link,
                &repo_at_home_link,
            ));
        }
        Action::FileMissing {
            repo_file,
            repo_class,
            would_be_home_file,
        } => {
            if matches!(repo_class, ValidFileType::File(_)) {
                handle!(handle_missing_file(
                    really_dry_run,
                    &would_be_home_file,
                    &repo_file
                ));
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

#[derive(Clone, Debug)]
enum Handled {
    Copy(Copy),
    Delete(PathBuf),
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
            Options::CopyFromRepoToHome,
            Options::CopyFromHomeToRepo,
            Options::HunksFromRepoToHome,
            Options::HunksFromHomeToRepo,
            Options::Meta(MetaDecision::Skip),
            Options::EditHomeAndRetry,
            Options::EditRepoAndRetry,
            Options::ShellAndRetry,
            Options::Meta(MetaDecision::Ignore),
            Options::Meta(MetaDecision::Abort),
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
            Options::Meta(MetaDecision::Skip),
            Options::ShellAndRetry,
            Options::Meta(MetaDecision::Ignore),
            Options::Meta(MetaDecision::Abort),
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
            Options::PullHomeAndRetry,
            Options::PullRepoAndRetry,
            Options::Meta(MetaDecision::Skip),
            Options::ShellAndRetry,
            Options::Meta(MetaDecision::Ignore),
            Options::Meta(MetaDecision::Abort),
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
            Options::CopyFromRepoToHome,
            Options::CopyFromHomeToRepo,
            Options::Meta(MetaDecision::Skip),
            Options::ShellAndRetry,
            Options::Meta(MetaDecision::Ignore),
            Options::Meta(MetaDecision::Abort),
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
            Options::CopyFromRepoToHome,
            Options::DeleteFromRepo,
            Options::Meta(MetaDecision::Skip),
            Options::ShellAndRetry,
            Options::Meta(MetaDecision::Ignore),
            Options::Meta(MetaDecision::Abort),
        ],
        really_dry_run,
        home_file,
        repo_file,
    )
}

fn make_decision<const N: usize>(
    options: [Options; N],
    really_dry_run: bool,
    home_file: impl AsRef<Path>,
    repo_file: impl AsRef<Path>,
) -> Result<Handled> {
    let decision = if really_dry_run {
        Options::Meta(MetaDecision::Skip)
    } else {
        let decision = selection_prompt(options, "What do you want to do?")?;
        let decision = decision.unwrap_or(Options::Meta(MetaDecision::Abort));

        println!("{decision}\n");

        decision
    };

    apply_decision(decision, home_file, repo_file)
}

fn apply_decision(
    decision: Options,
    home_file: impl AsRef<Path>,
    repo_file: impl AsRef<Path>,
) -> Result<Handled> {
    static EDITOR: LazyLock<OsString> =
        LazyLock::new(|| env::var_os("EDITOR").unwrap_or_else(|| "vi".into()));

    Ok(match decision {
        Options::Meta(meta) => Handled::Meta(meta),
        Options::CopyFromHomeToRepo => Handled::Copy(Copy {
            from: home_file.as_ref().to_path_buf(),
            to: repo_file.as_ref().to_path_buf(),
        }),
        Options::CopyFromRepoToHome => Handled::Copy(Copy {
            from: repo_file.as_ref().to_path_buf(),
            to: home_file.as_ref().to_path_buf(),
        }),
        Options::PullHomeAndRetry => {
            Command::new("git")
                .arg("pull")
                .current_dir(home_file.as_ref())
                .status()?;
            Handled::Meta(MetaDecision::Retry)
        }
        Options::PullRepoAndRetry => {
            Command::new("git")
                .arg("pull")
                .current_dir(repo_file.as_ref())
                .status()?;
            Handled::Meta(MetaDecision::Retry)
        }
        Options::EditHomeAndRetry => {
            Command::new(&*EDITOR).arg(home_file.as_ref()).status()?;
            Handled::Meta(MetaDecision::Retry)
        }
        Options::EditRepoAndRetry => {
            Command::new(&*EDITOR).arg(repo_file.as_ref()).status()?;
            Handled::Meta(MetaDecision::Retry)
        }
        Options::HunksFromRepoToHome => {
            let hunk = copy_hunks(repo_file.as_ref(), home_file.as_ref())?;
            Handled::Copy(Copy {
                from: hunk,
                to: home_file.as_ref().to_path_buf(),
            })
        }
        Options::HunksFromHomeToRepo => {
            let hunk = copy_hunks(home_file.as_ref(), repo_file.as_ref())?;
            Handled::Copy(Copy {
                from: hunk,
                to: repo_file.as_ref().to_path_buf(),
            })
        }
        Options::DeleteFromRepo => Handled::Delete(repo_file.as_ref().to_path_buf()),
        Options::ShellAndRetry => {
            let home_file = home_file.as_ref();
            let repo_file = repo_file.as_ref();
            let home_parent = home_file.parent();
            let repo_parent = repo_file.parent();

            let envs = [("HOME_FILE", home_file), ("REPO_FILE", repo_file)]
                .into_iter()
                .chain(home_parent.map(|p| ("HOME_PARENT", p)))
                .chain(repo_parent.map(|p| ("REPO_PARENT", p)));

            run_shell(envs)?;
            Handled::Meta(MetaDecision::Retry)
        }
    })
}

fn copy_hunks(from: &Path, to: &Path) -> Result<PathBuf> {
    fn temp_name() -> Result<OsString> {
        let mut buffer = [0u8; 4096];
        File::open("/dev/urandom")?.read_exact(&mut buffer)?;

        let mut hasher = DefaultHasher::new();
        hasher.write(&buffer);
        hasher.write_u32(std::process::id());
        if let Ok(unix) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            hasher.write_u64(unix.as_secs());
        }

        let hash = format!("{:08x}", hasher.finish());
        let hash = &hash[..8];

        let prefix = concat!(env!("CARGO_PKG_NAME"), "_git_hunks_");

        let cap = prefix.len() + hash.len();
        let mut buffer = OsString::with_capacity(cap);
        buffer.push(prefix);
        buffer.push(hash);

        Ok(buffer)
    }

    fn temp_dir() -> Result<PathBuf> {
        let temp_dir = env::temp_dir();

        for _ in 0..1024 {
            let name = temp_name()?;
            let path = temp_dir.join(name);

            match fs::create_dir(&path) {
                Ok(()) => return Ok(PathBuf::from_path_buf(path).expect("utf8 name")),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }

        Err(io::Error::other("failed to create temp dir").into())
    }

    fn temp_file(op: impl Fn(&Path) -> io::Result<()>) -> Result<PathBuf> {
        let temp_dir = env::temp_dir();

        for _ in 0..1024 {
            let name = temp_name()?;
            let path = temp_dir.join(name);
            let path = PathBuf::from_path_buf(path).expect("utf8 path");

            match op(&path) {
                Ok(()) => return Ok(path),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }

        Err(io::Error::other("failed to create temp file").into())
    }

    let from_buf;
    let from = if from.is_absolute() {
        from
    } else {
        from_buf = PathBuf::from_path_buf(from.canonicalize()?).expect("utf8 path");
        from_buf.as_path()
    };

    let to_buf;
    let to = if to.is_absolute() {
        to
    } else {
        to_buf = PathBuf::from_path_buf(to.canonicalize()?).expect("utf8 path");
        to_buf.as_path()
    };

    assert!(from.is_absolute(), "{from:?} must be absolute");
    assert!(to.is_absolute(), "{to:?} must be absolute");

    let git_dir = temp_dir()?;

    struct DropGuard<'a>(&'a Path);
    impl<'a> Drop for DropGuard<'a> {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.0);
        }
    }

    let git_dir = DropGuard(git_dir.as_path());

    let sh = xshell::Shell::new()?;
    sh.change_dir(git_dir.0);

    cmd!(sh, "git init").quiet().ignore_stdout().dbg_run()?;
    cmd!(sh, "git config user.name 'SyncDotfiles'")
        .quiet()
        .dbg_run()?;
    cmd!(sh, "git config user.email 'syncdotfiles@local.invalid'")
        .quiet()
        .dbg_run()?;

    let file_name = to.file_name().expect("file to have a file name");
    let git_file = git_dir.0.join(file_name);
    sh.copy_file(to, &git_file)?;

    cmd!(sh, "git add {file_name}").quiet().dbg_run()?;

    sh.copy_file(from, &git_file)?;

    Command::new("git")
        .args(["add", "--patch", file_name])
        .current_dir(git_dir.0)
        .status()?;

    cmd!(sh, "git restore {file_name}").quiet().dbg_run()?;

    let tmp = temp_file(|p| {
        fs::copy(&git_file, p)?;
        Ok(())
    })?;

    Ok(tmp)
}

fn run_shell<'x, 'y>(envs: impl IntoIterator<Item = (&'x str, &'y Path)>) -> Result<()> {
    static SHELL: LazyLock<OsString> =
        LazyLock::new(|| env::var_os("SHELL").unwrap_or_else(|| "sh".into()));

    Command::new(&*SHELL).envs(envs).status()?;
    Ok(())
}

#[derive(Clone, Debug)]
enum Options {
    Meta(MetaDecision),
    CopyFromHomeToRepo,
    CopyFromRepoToHome,
    PullHomeAndRetry,
    PullRepoAndRetry,
    EditHomeAndRetry,
    EditRepoAndRetry,
    HunksFromRepoToHome,
    HunksFromHomeToRepo,
    DeleteFromRepo,
    ShellAndRetry,
}

impl Display for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Meta(meta) => match meta {
                MetaDecision::Skip => write!(f, "Skip this file"),
                MetaDecision::Ignore => write!(f, "Ignore this file now and in the future"),
                MetaDecision::Abort => write!(f, "Skip all remaining files and abort"),
                MetaDecision::Retry => write!(f, "Just retry"),
            },
            Self::CopyFromRepoToHome => write!(f, "Copy the file from the repo to home"),
            Self::CopyFromHomeToRepo => write!(f, "Copy the file from home to the repo"),
            Self::PullHomeAndRetry => write!(f, "run `git pull` in home and retry"),
            Self::PullRepoAndRetry => write!(f, "run `git pull` in the repo and retry"),
            Self::EditHomeAndRetry => write!(f, "Edit the file in home and retry"),
            Self::EditRepoAndRetry => write!(f, "Edit the file in the repo and retry"),
            Self::HunksFromRepoToHome => write!(f, "Copy the file from the repo to home in hunks"),
            Self::HunksFromHomeToRepo => write!(f, "Copy the file from home to the repo in hunks"),
            Self::DeleteFromRepo => write!(f, "Delete the file from the repo"),
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
            Self::Meta(meta) => match meta {
                MetaDecision::Skip => 's',
                MetaDecision::Ignore => 'I',
                MetaDecision::Abort => 'Q',
                MetaDecision::Retry => '@',
            },
            Self::CopyFromRepoToHome => 'c',
            Self::CopyFromHomeToRepo => 'C',
            Self::PullHomeAndRetry => 'p',
            Self::PullRepoAndRetry => 'P',
            Self::EditHomeAndRetry => 'e',
            Self::EditRepoAndRetry => 'E',
            Self::HunksFromRepoToHome => 'h',
            Self::HunksFromHomeToRepo => 'H',
            Self::DeleteFromRepo => 'D',
            Self::ShellAndRetry => '$',
        }
    }

    fn also_matches(&self, &key: &Key) -> bool {
        matches!(
            (self, key),
            (Self::Meta(MetaDecision::Abort), Key::Char('q'))
                | (Self::Meta(MetaDecision::Ignore), Key::Char('i'))
                | (Self::Meta(MetaDecision::Skip), Key::Esc),
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

    let mut opts = OptItem::from(options).collect::<Vec<_>>();
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

    fn also_matches(&self, _key: &Key) -> bool {
        false
    }

    fn matches(&self, key: &Key) -> bool {
        *key == Key::Char(self.main_key()) || self.also_matches(key)
    }
}

impl<T: Display + ?Sized> SelectOption for (char, &T) {
    fn text(&self) -> &impl Display {
        &self.1
    }

    fn main_key(&self) -> char {
        self.0
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

#[derive(Debug)]
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
        let [home_hash, repo_hash] = hashes
            .lines()
            .collect::<Vec<_>>()
            .try_into()
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
    home: PathBuf,
    repo: PathBuf,
}

fn parse_args(home: PathBuf) -> Args {
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
        .fallback(1)
        .display_fallback();

    let home = short('H')
        .long("home")
        .argument::<PathBuf>("HOME")
        .help("The path to the home to sync")
        .fallback(home.to_path_buf())
        .debug_fallback();

    let repo = positional::<PathBuf>("REPO")
        .help("The path to the repo to sync")
        .fallback(PathBuf::from("."))
        .debug_fallback();

    let args = construct!(Args {
        dry_run,
        verbose,
        jobs,
        home,
        repo
    });

    args.to_options()
        .version(env!("CARGO_PKG_VERSION"))
        .descr("Synchronize dotfiles between home and the repo")
        .run()
}
