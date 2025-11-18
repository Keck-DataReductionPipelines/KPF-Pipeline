#!/usr/bin/env python3
import os
import argparse

from datetime import datetime
from collections import Counter
import compare_utils as utils


def main():
    from_addr = "kpfdrprun@kpfdrpserver"

    args = utils.parse_args()
    date_arg = args.date
    to_addr = args.email
    dir1_name = args.dir1
    dir2_name = args.dir2

    base1 = '/data/data_workspace',
    base2 = '/data/data_drp',
    base_dir = {'L0': base1, '2D': base1,  'L1': base2, 'L2': base2}

    dir1 = os.path.join(base_dir[dir1_name], dir1_name, date_arg)
    dir2 = os.path.join(base_dir[dir2_name], dir2_name, date_arg)

    print(f"Comparing {dir1} vs {dir2} OBJECT lists for {date_arg}")

    dir1_counts = utils.get_counts(dir1)
    dir2_counts = utils.get_counts(dir2)

    if not dir1_counts or not dir2_counts:
        msg = f"❌ Missing data in {dir1} or {dir2}"
        subj = f"[KPF DRP] Missing OBJECT data for {date_arg}"
        print(msg)
        utils.send_email(subj, msg, to_addr, from_addr)
        return

    missing = utils.compare_counts(dir1_counts, dir2_counts)

    dir1_cnt = Counter(dir1_counts)
    dir2_cnt = Counter(dir2_counts)

    if not missing:
        msg = f"✅ All OBJECTs in {dir1} are present in {dir2} for {date_arg}"
        print(msg)
    else:
        msg = utils.make_table(missing, date_arg, dir1_cnt, dir2_cnt, dir1_name, dir2_name)
        missing_files_table = utils.missing_filenames(missing, dir1, dir2, dir2_name)
        msg += "\n".join(missing_files_table)
        text_table = utils.html_to_text_table(msg)
        print(text_table)

    subj = f"[KPF DRP] {dir1_name} vs {dir2_name} OBJECT for {date_arg}"
    utils.send_email(subj, msg, to_addr, from_addr)


if __name__ == "__main__":
    main()

