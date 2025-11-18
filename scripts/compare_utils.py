import os
import smtplib
import argparse
import subprocess

from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
from collections import defaultdict
from email.message import EmailMessage


def parse_args():
    """
    Parse the command line arguments.  The date can be given as a command
    line argument.

    Returns: the parsed arguments object.

    """
    parser = argparse.ArgumentParser(
        description=f"Compare two levels of KPF OBJECTs for missing files."
    )
    parser.add_argument(
        "date", default=datetime.utcnow().strftime("%Y%m%d"),
        help="Date in YYYYMMDD format (default: today UTC)", nargs="?"
    )
    parser.add_argument(
        "--email", default="lfuhrman@keck.hawaii.edu",
        help="Email address to use."
    )
    parser.add_argument(
        "--dir1", default="L0", choices=["L0", "2D", "L1", "L2"],
        help="Processing Level to compare (allowed: L0, 2D, L1, L2)."
    )
    parser.add_argument(
        "--dir2", default="L2", choices=["L0", "2D", "L1", "L2"],
        help="Processing Level to compare (allowed: L0, 2D, L1, L2)."
    )

    args = parser.parse_args()

    return args


def html_to_text_table(html_content):
    """
    parse the html (used by the email) content into a text table.

    Args:
        html_content ():

    Returns:

    """
    soup = BeautifulSoup(html_content, "html.parser")
    lines = []

    # Process the tables separately
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            row = []
            for cell in tr.find_all(["td", "th"]):
                row.append(cell.get_text(strip=True))
            if row:
                lines.append(" | ".join(f"{col:<25}" for col in row))

        lines.append("")

    return "\n".join(lines)



def missing_filenames(missing, dir1, dir2, dir2_name):
    """
    Finds the missing filenames between DIR1 and DIR2 for non-calibcations.
    Calibrations are defined as 'autocal' or are in the caltypes list.

    Args:
        missing ():
        dir1 ():
        dir2 ():

    Returns:

    """

    cal_types = (
        'autocal-bias', 'autocal-bias', 'flush-persistence', 'autocal-dark',
        'slewcal', 'autocal-thar-hk', 'autocal-thar-all-morn'
    )
    dir1_filenames = get_filenames(dir1)
    dir2_filenames = get_filenames(dir2)

    missing_files_table = [
        f"<h3>Missing {dir2_name} Files (non-calibrations)</h3>",
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse: collapse;'>",
        f"<tr><th>OBJECT</th><th>Missing {dir2_name} Files </th></tr>"
    ]

    for obj_type in missing:
        if obj_type.lower() in cal_types or 'autocal' in obj_type.lower():
            continue

        # Get missing files
        files_missing = find_missing_files(obj_type, dir1_filenames,
                                           dir2_filenames, dir2_name)
        if not files_missing:
            files_missing_str = "None"
        else:
            files_missing_str = ", ".join(files_missing)

        # Add row
        missing_files_table.append(f"<tr><td>{obj_type}</td><td>"
                                   f"{files_missing_str}</td></tr>")

    missing_files_table.append("</table>")

    return missing_files_table


def get_filenames(direct):
    """
    Return a dict mapping OBJECT -> list of filenames in the directory.
    If OFNAME is missing, uses None. Works for all FITS files in the directory.
    """
    if not os.path.isdir(direct):
        return {}

    try:
        # Grab OBJECT and OFNAME lines, skip GCOFNAME
        cmd = (f"/usr/local/anaconda3/bin/fitsheader {direct}/*.fits 2>/dev/null "
               f"| grep -E 'OBJECT|OFNAME' | grep -v GCOFNAME")
        output = subprocess.check_output(cmd, shell=True, text=True)
    except subprocess.CalledProcessError:
        return {}

    filenames = defaultdict(list)
    current_fname = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("OFNAME"):
            parts = line.split("'")
            current_fname = parts[1].strip() if len(parts) > 1 else None
        elif line.startswith("OBJECT"):
            parts = line.split("'")
            current_obj = parts[1].strip() if len(parts) > 1 else None

            if current_obj:
                filenames[current_obj].append(current_fname)
            current_fname = None

    return dict(filenames)


def find_missing_files(obj, filenames_dir1, filenames_dir2, dir2_name):
    """
    Filenames are OFNAME,  original KOA name thus no trailing _DIR1 or _DIR2
    Args:
        obj (): The object name,  ie T006038
        filenames_dir1 (): A list of the dir1 filenames from the OFNAME keyword
        filenames_dir2 (): A list of the dir2 filenames from the OFNAME keyword

    Returns:

    """
    missing_files = []
    l2_set = set(filenames_dir2.get(obj, []))

    for fname in filenames_dir1.get(obj, []):
        if fname and fname not in l2_set:
            missing_files.append(fname.replace(".fits", f"_{dir2_name}.fits"))

    return missing_files


def compare_counts(counts_dir1, counts_dir2):
    """Compare OBJECT lists between DIR1 and DIR2."""
    missing_in_l2 = {}
    for obj, count in counts_dir1.items():
        if obj not in counts_dir2:
            missing_in_l2[obj] = count
        elif count > counts_dir2[obj]:
            missing_in_l2[obj] = count - counts_dir2[obj]
    return missing_in_l2


def get_counts(direct):
    """
    Return a dict {OBJECT: count} for all FITS files in a directory.
    Uses 'fitsheader -k OBJECT -f' for robust parsing.
    Cleans extra spaces or repeated OBJECT strings.
    """
    if not os.path.isdir(direct):
        return None

    cmd = f"/usr/local/anaconda3/bin/fitsheader -k OBJECT -f {direct}/*.fits 2>/dev/null"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
    except subprocess.CalledProcessError:
        return {}

    objects = Counter()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("filename") or line.startswith("---"):
            continue  # skip header lines

        # Split into filename and object
        parts = line.split(None, 1)
        if len(parts) == 2:
            obj = parts[1].strip()

            # OBJECT can be repeated or padded: take only first part or strip
            obj_clean = obj.split()[0] if '--' not in obj else obj.split('--')[0].strip()
            objects[obj_clean] += 1

    return dict(objects)



def make_table(missing, date_arg, dir1_cnt, dir2_cnt, dir1_name, dir2_name):
    msg_lines = [
        f"<h3>Missing in {dir2_name} for {date_arg}</h3>",
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse: collapse;'>",
        f"<tr><th>OBJECT</th><th>{dir1_name}</th><th>{dir2_name}</th>"
        f"<th>MISSING</th></tr>"
    ]

    total_missing = 0

    expected_missing = [
        f"<h3>Expected Cals Missing in {dir2_name} for {date_arg}</h3>",
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse: collapse;'>",
        f"<tr><th>OBJECT</th><th>{dir1_name}</th><th>{dir2_name}</th>"
        f"<th>MISSING</th></tr>"
    ]

    expected_cnt = 0

    for obj, diff in sorted(missing.items(), key=lambda x: -x[1]):

        row = (f"<tr><td>{obj}</td><td>{dir1_cnt[obj]}</td><td>{dir2_cnt[obj]}"
               f"</td><td>{diff}</td></tr>")

        # Expected categories
        if 'bias' in obj or 'flush' in obj or 'dark' in obj:
            expected_missing.append(row)
            expected_cnt += diff
            continue

        # Not expected missing
        msg_lines.append(row)
        total_missing += diff

    msg_lines.append("</table>")
    msg_lines.append(f"<p><b>Total NOT expected missing OBJECTs: {total_missing}</b></p>")

    expected_missing.append("</table>")
    expected_missing.append(f"<p><b>Total expected missing:</b></p>")

    msg = "\n".join(msg_lines) + "\n" + "\n".join(expected_missing)

    return msg


def send_email(subject, body_html, to_addr, from_addr):
    """
    Send an HTML email via localhost.
    """
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"]   = to_addr
    msg["Subject"] = subject

    # Catch for non-html email clients
    msg.set_content("HTML content not displayed. "
                    "Please use an HTML-capable email client.")

    msg.add_alternative(body_html, subtype="html")

    with smtplib.SMTP("localhost") as s:
        s.send_message(msg)

