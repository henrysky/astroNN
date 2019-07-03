"""
Pure python patching with brute-force line-by-line non-recursive parsing 
Original code adapted from Copyright (c) 2008-2016 Anatoly Techtonik <techtonik@gmail.com>
with MIT license
"""
import copy
import logging
import re

from os.path import exists, isfile, abspath
import os
import shutil

# Logging is controlled by logger named after the module name
logger = logging.getLogger(__name__)
debug = logger.debug
info = logger.info
warning = logger.warning
# initialize logger itself and add a NulHandler
# https://docs.python.org/3.3/howto/logging.html#configuring-logging-for-a-library
logger.addHandler(logging.NullHandler())


class Hunk(object):
    """
    Parsed hunk data container (hunk starts with @@ -R +R @@)
    """
    def __init__(self):
        self.startsrc = None  #: line count starts with 1
        self.linessrc = None
        self.starttgt = None
        self.linestgt = None
        self.invalid = False
        self.desc = ''
        self.text = []


class _Patch(object):
    """ Patch for a single file.
        If used as an iterable, returns hunks.
    """
    def __init__(self):
        self.source = None
        self.target = None
        self.hunks = []
        self.hunkends = []
        self.header = []

        self.type = None

    def __iter__(self):
        for h in self.hunks:
            yield h


class Patch(object):
    """
    Patch is a patch parser and container.
    When used as an iterable, returns patches.
    """
    def __init__(self, patchpath=None):
        # name of the PatchSet (filepath or ...)
        self.name = None
        # patch set type - one of constants
        self.type = None

        # list of Patch objects
        self.items = []

        self.errors = 0  # fatal parsing errors
        self.warnings = 0  # non-critical warnings

        with open(patchpath, "rb") as fp:  # parse .patch or .diff file
            self.parse(fp)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i in self.items:
            yield i

    def parse(self, stream):
        """
        parse unified diff
        return True on success
        """
        lineends = dict(lf=0, crlf=0, cr=0)
        nexthunkno = 0  #: even if index starts with 0 user messages number hunks from 1

        p = None
        hunk = None
        # hunkactual variable is used to calculate hunk lines for comparison
        hunkactual = dict(linessrc=None, linestgt=None)

        class wrapumerate(enumerate):
            """Enumerate wrapper that uses boolean end of stream status instead of
            StopIteration exception, and properties to access line information.
            """

            def __init__(self, *args, **kwargs):
                # we don't call parent, it is magically created by __new__ method

                self._exhausted = False
                self._lineno = False  # after end of stream equal to the num of lines
                self._line = False  # will be reset to False after end of stream

            def next(self):
                """Try to read the next line and return True if it is available,
                   False if end of stream is reached."""
                if self._exhausted:
                    return False

                try:
                    self._lineno, self._line = super(wrapumerate, self).__next__()
                except StopIteration:
                    self._exhausted = True
                    self._line = False
                    return False
                return True

            @property
            def is_empty(self):
                return self._exhausted

            @property
            def line(self):
                return self._line

            @property
            def lineno(self):
                return self._lineno

        # define states (possible file regions) that direct parse flow
        headscan = True  # start with scanning header
        filepaths = False  # lines starting with --- and +++

        hunkhead = False  # @@ -R +R @@ sequence
        hunkbody = False  #
        hunkskip = False  # skipping invalid hunk mode

        hunkparsed = False  # state after successfully parsed hunk

        # regexp to match start of hunk, used groups - 1,3,4,6
        re_hunk_start = re.compile(b"^@@ -(\d+)(,(\d+))? \+(\d+)(,(\d+))? @@")

        self.errors = 0
        # temp buffers for header and filepaths info
        header = []
        srcname = None
        tgtname = None

        # start of main cycle
        # each parsing block already has line available in fe.line
        fe = wrapumerate(stream)
        while fe.next():

            # -- deciders: these only switch state to decide who should process
            # --           line fetched at the start of this cycle
            if hunkparsed:
                hunkparsed = False
                if re_hunk_start.match(fe.line):
                    hunkhead = True
                elif fe.line.startswith(b"--- "):
                    filepaths = True
                else:
                    headscan = True
            # -- ------------------------------------

            # read out header
            if headscan:
                while not fe.is_empty and not fe.line.startswith(b"--- "):
                    header.append(fe.line)
                    fe.next()
                if fe.is_empty:
                    if p is None:
                        debug("no patch data found")  # error is shown later
                        self.errors += 1
                    else:
                        info(f"{len(b''.join(header))} unparsed bytes left at the end of stream")
                        self.warnings += 1
                        # otherwise error += 1
                    # this is actually a loop exit
                    continue

                headscan = False
                # switch to filepaths state
                filepaths = True

            line = fe.line
            lineno = fe.lineno

            # hunkskip and hunkbody code skipped until definition of hunkhead is parsed
            if hunkbody:
                # [x] treat empty lines inside hunks as containing single space
                #     (this happens when diff is saved by copy/pasting to editor
                #      that strips trailing whitespace)
                if line.strip(b"\r\n") == b"":
                    debug("expanding empty line in a middle of hunk body")
                    self.warnings += 1
                    line = b' ' + line

                # process line first
                if re.match(b"^[- \\+\\\\]", line):
                    # gather stats about line endings
                    if line.endswith(b"\r\n"):
                        p.hunkends["crlf"] += 1
                    elif line.endswith(b"\n"):
                        p.hunkends["lf"] += 1
                    elif line.endswith(b"\r"):
                        p.hunkends["cr"] += 1

                    if line.startswith(b"-"):
                        hunkactual["linessrc"] += 1
                    elif line.startswith(b"+"):
                        hunkactual["linestgt"] += 1
                    elif not line.startswith(b"\\"):
                        hunkactual["linessrc"] += 1
                        hunkactual["linestgt"] += 1
                    hunk.text.append(line)
                else:
                    warning(f"invalid hunk no.{nexthunkno} at {lineno + 1} for target file {p.target}")
                    # add hunk status node
                    hunk.invalid = True
                    p.hunks.append(hunk)
                    self.errors += 1
                    # switch to hunkskip state
                    hunkbody = False
                    hunkskip = True

                # check exit conditions
                if hunkactual["linessrc"] > hunk.linessrc or hunkactual["linestgt"] > hunk.linestgt:
                    warning(f"extra lines for hunk no.{nexthunkno} at {lineno + 1} for target {p.target}")
                    # add hunk status node
                    hunk.invalid = True
                    p.hunks.append(hunk)
                    self.errors += 1
                    # switch to hunkskip state
                    hunkbody = False
                    hunkskip = True
                elif hunk.linessrc == hunkactual["linessrc"] and hunk.linestgt == hunkactual["linestgt"]:
                    # hunk parsed successfully
                    p.hunks.append(hunk)
                    # switch to hunkparsed state
                    hunkbody = False
                    hunkparsed = True

                    # detect mixed window/unix line ends
                    ends = p.hunkends
                    if ((ends["cr"] != 0) + (ends["crlf"] != 0) + (ends["lf"] != 0)) > 1:
                        warning(f"inconsistent line ends in patch hunks for {p.source}")
                        self.warnings += 1
                    # fetch next line
                    continue

            if hunkskip:
                if re_hunk_start.match(line):
                    # switch to hunkhead state
                    hunkskip = False
                    hunkhead = True
                elif line.startswith(b"--- "):
                    # switch to filepaths state
                    hunkskip = False
                    filepaths = True

            if filepaths:
                if line.startswith(b"--- "):
                    if srcname is not None:
                        # XXX testcase
                        warning(f"skipping false patch for {srcname}")
                        srcname = None
                        # XXX header += srcname
                        # double source filepath line is encountered
                        # attempt to restart from this second line
                    re_filepath = b"^--- ([^\t]+)"
                    match = re.match(re_filepath, line)
                    if match:
                        srcname = match.group(1).strip()
                    else:
                        warning(f"skipping invalid filepath at line {lineno + 1}")
                        self.errors += 1
                        # XXX p.header += line
                        # switch back to headscan state
                        filepaths = False
                        headscan = True
                elif not line.startswith(b"+++ "):
                    if srcname is not None:
                        warning(f"skipping invalid patch with no target for {srcname}")
                        self.errors += 1
                        srcname = None
                        # XXX header += srcname
                        # XXX header += line
                    else:
                        # this should be unreachable
                        warning("skipping invalid target patch")
                    filepaths = False
                    headscan = True
                else:
                    if tgtname is not None:
                        # XXX seems to be a dead branch
                        warning(f"skipping invalid patch - double target at line {lineno + 1}")
                        self.errors += 1
                        srcname = None
                        tgtname = None
                        # XXX header += srcname
                        # XXX header += tgtname
                        # XXX header += line
                        # double target filepath line is encountered
                        # switch back to headscan state
                        filepaths = False
                        headscan = True
                    else:
                        re_filepath = b"^\+\+\+ ([^\t]+)"
                        match = re.match(re_filepath, line)
                        if not match:
                            warning(f"skipping invalid patch - no target filepath at line {lineno + 1}")
                            self.errors += 1
                            srcname = None
                            # switch back to headscan state
                            filepaths = False
                            headscan = True
                        else:
                            if p:  # for the first run p is None
                                self.items.append(p)
                            p = _Patch()
                            p.source = srcname
                            srcname = None
                            p.target = match.group(1).strip()
                            p.header = header
                            header = []
                            # switch to hunkhead state
                            filepaths = False
                            hunkhead = True
                            nexthunkno = 0
                            p.hunkends = lineends.copy()
                            continue

            if hunkhead:
                match = re.match(b"^@@ -(\d+)(,(\d+))? \+(\d+)(,(\d+))? @@(.*)", line)
                if not match:
                    if not p.hunks:
                        warning(f"skipping invalid patch with no hunks for file {p.source}")
                        self.errors += 1
                        # XXX review switch
                        # switch to headscan state
                        hunkhead = False
                        headscan = True
                        continue
                    else:
                        # switch to headscan state
                        hunkhead = False
                        headscan = True
                else:
                    hunk = Hunk()
                    hunk.startsrc = int(match.group(1))
                    hunk.linessrc = 1
                    if match.group(3): hunk.linessrc = int(match.group(3))
                    hunk.starttgt = int(match.group(4))
                    hunk.linestgt = 1
                    if match.group(6): hunk.linestgt = int(match.group(6))
                    hunk.invalid = False
                    hunk.desc = match.group(7)[1:].rstrip()
                    hunk.text = []

                    hunkactual["linessrc"] = hunkactual["linestgt"] = 0

                    # switch to hunkbody state
                    hunkhead = False
                    hunkbody = True
                    nexthunkno += 1
                    continue

        # /while fe.next()

        if p:
            self.items.append(p)

        if not hunkparsed:
            if hunkskip:
                warning("warning: finished with errors, some hunks may be invalid")
            elif headscan:
                if len(self.items) == 0:
                    warning("error: no patch data found!")
                    return False
                else:  # extra data at the end of file
                    pass
            else:
                warning("error: patch stream is incomplete!")
                self.errors += 1
                if len(self.items) == 0:
                    return False

        # XXX fix total hunks calculation
        debug(f"total files: {len(self.items)} total hunks: {sum(len(p.hunks))}" for p in self.items)

        # ---- detect patch and patchset types ----
        for idx, p in enumerate(self.items):
            self.items[idx].type = 'git'

        types = set([p.type for p in self.items])
        if len(types) > 1:
            self.type = 'mixed'
        else:
            self.type = types.pop()
        # --------

        return self.errors == 0

    def apply(self, filepath=None):
        """ Apply parsed patch, optionally stripping leading components
            from file paths. `root` parameter specifies working dir.
            return True on success
        """

        total = len(self.items)
        errors = 0

        # for fileno, filepath in enumerate(self.source):
        for i, p in enumerate(self.items):
            if not isfile(filepath):
                warning(f"not a file: {filepath}")
                errors += 1
                continue

            # [ ] check absolute paths security here
            debug(f"processing {i + 1}/{total}:\t {filepath}")

            # validate before patching
            f2fp = open(filepath, 'rb')
            hunkno = 0
            hunk = p.hunks[hunkno]
            hunkfind = []
            validhunks = 0
            canpatch = False
            for lineno, line in enumerate(f2fp):
                if lineno + 1 < hunk.startsrc:
                    continue
                elif lineno + 1 == hunk.startsrc:
                    hunkfind = [x[1:].rstrip(b"\r\n") for x in hunk.text if x[0] in b" -"]
                    hunklineno = 0

                # check hunks in source file
                if lineno + 1 < hunk.startsrc + len(hunkfind) - 1:
                    if line.rstrip(b"\r\n") == hunkfind[hunklineno]:
                        hunklineno += 1
                    else:
                        info("file %d/%d:\t %s" % (i + 1, total, filepath))
                        info(f" hunk no.{hunkno + 1} doesn't match source file at line {lineno + 1}")
                        info(f"  expected: {hunkfind[hunklineno]}")
                        info(f"  actual  : %s" % line.rstrip(b"\r\n"))
                        # not counting this as error, because file may already be patched.
                        # check if file is already patched is done after the number of
                        # invalid hunks if found
                        # continue to check other hunks for completeness
                        hunkno += 1
                        if hunkno < len(p.hunks):
                            hunk = p.hunks[hunkno]
                            continue
                        else:
                            break

                # check if processed line is the last line
                if lineno + 1 == hunk.startsrc + len(hunkfind) - 1:
                    debug(" hunk no.%d for file %s  -- is ready to be patched" % (hunkno + 1, filepath))
                    hunkno += 1
                    validhunks += 1
                    if hunkno < len(p.hunks):
                        hunk = p.hunks[hunkno]
                    else:
                        if validhunks == len(p.hunks):
                            # patch file
                            canpatch = True
                            break
            else:
                if hunkno < len(p.hunks):
                    warning("premature end of source file %s at hunk %d" % (filepath, hunkno + 1))
                    errors += 1

            f2fp.close()

            if validhunks < len(p.hunks):
                if self._match_file_hunks(filepath, p.hunks):
                    warning(f"already patched {filepath}")
                else:
                    warning(f"source file is different - {filepath}")
                    errors += 1
            if canpatch:
                backupname = filepath + ".orig"
                if exists(backupname):
                    warning(f"can't backup original file to {backupname} - aborting")
                else:
                    shutil.move(filepath, backupname)
                    if self.write_hunks(backupname, filepath, p.hunks):
                        info(f"successfully patched {i + 1}/{total}:\t {filepath}")
                        os.unlink(backupname)
                    else:
                        errors += 1
                        warning(f"error patching file {filepath}")
                        shutil.copy(filepath, filepath + ".invalid")
                        warning(f"invalid version is saved to {filepath}.invalid")
                        shutil.move(backupname, filepath)

        return errors

    def _reverse(self):
        """ reverse patch direction (this doesn't touch filepaths) """
        for p in self.items:
            for h in p.hunks:
                h.startsrc, h.starttgt = h.starttgt, h.startsrc
                h.linessrc, h.linestgt = h.linestgt, h.linessrc
                for i, line in enumerate(h.text):
                    # need to use line[0:1] here, because line[0]
                    # returns int instead of bytes on Python 3
                    if line[0:1] == b'+':
                        h.text[i] = b'-' + line[1:]
                    elif line[0:1] == b'-':
                        h.text[i] = b'+' + line[1:]

    def revert(self, filepath=None):
        """ apply patch in reverse order """
        reverted = copy.deepcopy(self)
        reverted._reverse()
        return reverted.apply(filepath)

    def _match_file_hunks(self, filepath, hunks):
        matched = True
        fp = open(abspath(filepath), 'rb')

        class NoMatch(Exception):
            pass

        lineno = 1
        line = fp.readline()
        try:
            for hno, h in enumerate(hunks):
                # skip to first line of the hunk
                while lineno < h.starttgt:
                    if not len(line):  # eof
                        debug(f"check failed - premature eof before hunk: {hno + 1}")
                        raise NoMatch
                    line = fp.readline()
                    lineno += 1
                for hline in h.text:
                    if hline.startswith(b"-"):
                        continue
                    if not len(line):
                        debug(f"check failed - premature eof on hunk: {hno + 1}")
                        raise NoMatch
                    if line.rstrip(b"\r\n") != hline[1:].rstrip(b"\r\n"):
                        debug("file is not patched - failed hunk: %d" % (hno + 1))
                        raise NoMatch
                    line = fp.readline()
                    lineno += 1

        except NoMatch:
            matched = False

        fp.close()
        return matched

    def patch_stream(self, instream, hunks):
        """ Generator that yields stream patched with hunks iterable

            Converts lineends in hunk lines to the best suitable format
            autodetected from input
        """
        hunks = iter(hunks)
        srclineno = 1
        lineends = {b'\n': 0, b'\r\n': 0, b'\r': 0}

        def get_line():
            """
            local utility function - return line from source stream
            collecting line end statistics on the way
            """
            line = instream.readline()
            # 'U' mode works only with text files
            if line.endswith(b"\r\n"):
                lineends[b"\r\n"] += 1
            elif line.endswith(b"\n"):
                lineends[b"\n"] += 1
            elif line.endswith(b"\r"):
                lineends[b"\r"] += 1
            return line

        for hno, h in enumerate(hunks):
            debug(f"hunk {hno + 1}")
            # skip to line just before hunk starts
            while srclineno < h.startsrc:
                yield get_line()
                srclineno += 1

            for hline in h.text:
                if hline.startswith(b"-") or hline.startswith(b"\\"):
                    get_line()
                    srclineno += 1
                    continue
                else:
                    if not hline.startswith(b"+"):
                        get_line()
                        srclineno += 1
                    line2write = hline[1:]
                    # detect if line ends are consistent in source file
                    if sum([bool(lineends[x]) for x in lineends]) == 1:
                        newline = [x for x in lineends if lineends[x] != 0][0]
                        yield line2write.rstrip(b"\r\n") + newline
                    else:  # newlines are mixed
                        yield line2write

        for line in instream:
            yield line

    def write_hunks(self, srcname, tgtname, hunks):
        src = open(srcname, "rb")
        tgt = open(tgtname, "wb")

        debug(f"processing target file {tgtname}")

        tgt.writelines(self.patch_stream(src, hunks))

        tgt.close()
        src.close()
        shutil.copymode(srcname, tgtname)
        return True
