import os
import re
import magic

mime = magic.Magic(mime=True)


def traverse(top, sort_key=None, contains=None, re_match=None, mime_type=None):
    """
     traverse the input directory
    :param top:
    :param sort_key:
    :param contains:
    :param re_match:
    :param mime_type:
    :return:
    """
    input_files = list()
    if os.path.isfile(top):
        input_files.append(top)
        return input_files

    pattern = None
    if re_match:
        pattern = re.compile(re_match)

    for root, dirs, files in os.walk(top):
        for filename in files:
            file_path = os.path.join(root, filename)
            # if (contains is None and re_match is None) or \
            #         (pattern and pattern.match(filename)) or \
            #         (contains and contains in filename):
            if contains and contains not in filename:
                continue
            if re_match and not pattern.match(filename):
                continue
            if mime_type and mime.from_file(file_path) != mime_type:
                continue

            input_files.append(file_path)
    print("File total count: %d" % len(input_files))
    input_files = sorted(input_files, key=sort_key)
    return input_files


def reader_g(top, prefix=None, suffix=None, strip="\r\n ", skip_blank=True, skip_head=0):
    """
    file reader generator
    :param top:
    :param prefix:
    :param suffix:
    :param strip:
    :param skip_blank: ignore blank line or not
    :param skip_head: skip the corresponding line of header
    :return:
    """
    filename_pattern = r""
    if prefix:
        filename_pattern += "^" + prefix + ".*"
    if suffix:
        filename_pattern += ".*" + suffix + "$"

    files = traverse(top, re_match=filename_pattern)
    for path in files:
        with open(path, "rb") as fopen:
            print("Load: '%s'" % path)
            count = 0
            while True:
                line = fopen.readline()
                if not line:
                    break
                # skip head line
                count += 1
                if count <= skip_head:
                    continue
                # check the line whether is blank or not
                line = line.strip(strip)
                if skip_blank and not line:
                    continue
                yield line
