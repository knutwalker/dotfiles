#!/usr/bin/env python

from __future__ import print_function
import urlparse

def parse_url(url):
    if not url.startswith('http'):
        url = 'http://' + url
    return urlparse.urlparse(url)

def clean_path(parts):
    dp_index = parts.index('dp')
    if dp_index is -1:
        return None
    return '/'.join(parts[dp_index:dp_index + 2] + [''])

def clean_url(url):
    parts = url.path.split('/')
    path = clean_path(parts)
    if path is None:
        return url
    return url._replace(path=path, params='', query='', fragment='')

def clean_link(link):
    url = parse_url(link)
    url = clean_url(url)
    return url.geturl()

if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--alfred":
        from xml.sax.saxutils import escape
        link = escape(clean_link(sys.argv[2]))
        print("""<?xml version="1.0"?>
<items>
    <item uid="amznlink" arg="{0}">
        <title>{0}</title>
        <subtitle>Press Enter or Cmd+C to copy</subtitle>
        <icon>icon.png</icon>
    </item>
</items>""".format(link))
    else:
        print(clean_link(sys.argv[1]))
