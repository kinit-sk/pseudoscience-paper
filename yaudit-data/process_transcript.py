import re


def remove_tags(text):
    """
    Remove vtt markup tags
    """
    tags = [
        r'</c>',
        r'<c(\.color\w+)?>',
        r'<\d{2}:\d{2}:\d{2}\.\d{3}>',

    ]

    for pat in tags:
        text = re.sub(pat, '', text)

    # extract timestamp, only kep HH:MM
    text = re.sub(
        r'(\d{2}:\d{2}):\d{2}\.\d{3} --> .* align:start position:0%',
        r'\g<1>',
        text
    )

    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    return text

def remove_header(lines):
    """
    Remove vtt file header
    """
    pos = -1
    for mark in ('##', 'Language: en',):
        if mark in lines:
            pos = lines.index(mark)
    lines = lines[pos+1:]
    return lines


def merge_duplicates(lines):
    """
    Remove duplicated subtitles. Duplacates are always adjacent.
    """
    last_timestamp = ''
    last_cap = ''
    for line in lines:
        if line == "":
            continue
        if re.match('^\d{2}:\d{2}$', line):
            if line != last_timestamp:
#                 yield line
                last_timestamp = line
        else:
            if line != last_cap:
                yield line
                last_cap = line


def merge_short_lines(lines):
    buffer = ''
    for line in lines:
        if line == "" or re.match('^\d{2}:\d{2}$', line):
            yield '\n' + line
            continue

        if len(line+buffer) < 80:
            buffer += ' ' + line
        else:
            yield buffer.strip()
            buffer = line
    yield buffer


def parse_transcript(text):
    text = remove_tags(text)
    lines = text.splitlines()
    lines = remove_header(lines)
    lines = merge_duplicates(lines)
    lines = list(lines)
    lines = merge_short_lines(lines)
    lines = list(lines)
    lines = [
        re.sub('\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', '', line).strip()
        for line in lines
    ]
    return [
        l for l in lines if l
    ]
