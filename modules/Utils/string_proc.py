from datetime import datetime, timedelta
from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext


class str_replace(KPF_Primitive):
    """
    This primitive does string replacement

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `str_replace` event issued in the recipe:

            - `action.args[0] (string)`: string with old value to be replaced with new value.
            - `action.args[1] (string)`: old value string to be replaced.
            - `action.args[2] (string)`: new value replacement
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)

    def _pre_condition(self) -> bool:
        success = len(self.action.args) >= 3 and isinstance(self.action.args[0], str) and \
                  isinstance(self.action.args[1], str) and isinstance(self.action.args[2], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        original_value = self.action.args[0]
        old_value = self.action.args[1]
        new_value = self.action.args[2]
        new_string = original_value.replace(old_value, new_value)
        return Arguments(new_string)


class date_from_kpffile(KPF_Primitive):
    """
    This primitive determines the datecode (YYYYMMDD) of a KPF file.

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `date_from_kpffile` event issued in the recipe:

            - `action.args[0] (string)`: filename in kpf format
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        if 'startswith' in args_keys and action.args['startswith']:
            self.filenamestartwith = action.args['startswith']
        else:
            self.filenamestartwith = ''

        self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        success = len(self.action.args) == 1 and isinstance(self.action.args[0], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        f_name = self.action.args[0]
        first_key = self.filenamestartwith or 'KP.'
        date_format = 'YYYYMMDD'
        first_idx = f_name.find(first_key)
        date_str = None
        if first_idx >= 0:
            start_idx = first_idx + len(first_key)
            date_str = f_name[start_idx:start_idx+len(date_format)]

        if not date_str:
            if self.logger:
                self.logger.info("can not find date from file name, " + f_name)
        else:
            if self.logger:
                self.logger.info("File is from " + date_str)

        return Arguments(date_str)


class days_since_observation(KPF_Primitive):
    """
    This primitive returns the number of days (float) since the time encoded
    in the ObsID of a KPF filename (e.g., 'KP.20240101.10214.08.fits' or
    '/data/L1/20240101/KP.20240101.10214.08_L1.fits')

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `days_since_observation` event issued in the recipe:

            - `action.args[0] (float)`: days since time in encoded in filename
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        if 'startswith' in args_keys and action.args['startswith']:
            self.filenamestartwith = action.args['startswith']
        else:
            self.filenamestartwith = ''

        self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        success = len(self.action.args) == 1 and isinstance(self.action.args[0], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        f_name = self.action.args[0]
        first_key = self.filenamestartwith or 'KP.'
        date_format = 'YYYYMMDD.SSSSS.SS'
        first_idx = f_name.find(first_key)
        days_since = 0
        if first_idx >= 0:
            start_idx = first_idx + len(first_key)
            date_str = f_name[start_idx:start_idx+len(date_format)]
            date_parts = date_str.split('.')
            if len(date_parts) == 3:
                try:
                    yyyymmdd = date_parts[0]
                    year = yyyymmdd[0:4]
                    month = yyyymmdd[4:6]
                    day = yyyymmdd[6:8]
                    sec = date_parts[1]
                    then = datetime(int(year), int(month), int(day)) + timedelta(seconds=int(sec))
                    now = datetime.now()
                    days_since = (now - then).total_seconds() / 86400
                except:
                    self.logger.info("Error computing time difference.")

        if days_since == 0:
            if self.logger:
                self.logger.info("Can not find date from " + f_name)
        else:
            if self.logger:
                self.logger.info(f"Days since observation of {f_name}: " + str(days_since))

        return Arguments(days_since)


class date_from_path(KPF_Primitive):
    """
    This primitive determines the datecode (YYYYMMDD) from a path
    (e.g., '/data/masters/20230711/kpf_20230711_master_arclamp_autocal-etalon-all-eve.fits').

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `date_from_path` event issued in the recipe:

            - `action.args[0] (string)`: path
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        success = len(self.action.args) == 1 and isinstance(self.action.args[0], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        # This algorithm finds the first 8-digit string and returns it.
        path = self.action.args[0]
        digit_count = 0
        potential_match = ""
        for char in path:
            if char.isdigit():
                digit_count += 1
                potential_match += char
                if digit_count == 8:
                    return Arguments(potential_match)
            else:
                digit_count = 0
                potential_match = ""

        return Arguments('99999999') # Error code indicating that no match was found


class level_from_kpffile(KPF_Primitive):
    """
    This primitive determines the KPF data level (L0, 2D, L1, L2, None) of a KPF file.

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains
                  positional arguments and  keyword arguments passed by the
                  `level_from_kpffile` event issued in the recipe:

            - `action.args[0] (string)`: filename in kpf format
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        success = len(self.action.args) == 1 and isinstance(self.action.args[0], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        fullpath = self.action.args[0]  # e.g. '/data/L0/20230724/KP.20230720.12345.67.fits'
        filename = fullpath.rsplit('/', 1)[-1]  # e.g., 'KP.20230720.12345.67.fits'
        format_type = None
        # Check that the filename has the correct length
        if len(filename) in [25, 28, 28, 28]: # L0, 2D, L1, L2
            # Check each part of the filename
            parts = filename.split('.')
            if len(parts) == 5 and parts[0] == 'KP' and parts[1].isdigit() and len(parts[1]) == 8 \
                    and parts[2].isdigit() and len(parts[2]) == 5:
                suffix = parts[3].split('_')
                if len(suffix) == 1 and suffix[0][:2].isdigit() and len(suffix[0][:2]) == 2:
                    format_type = 'L0'
                elif len(suffix) == 2 and suffix[0][:2].isdigit() and len(suffix[0][:2]) == 2 \
                        and suffix[1] in ['2D', 'L1', 'L2']:
                    format_type = suffix[1]
        else:
            format_type = None

        return Arguments(format_type) # returns 'L0', '2D', 'L1', 'L2', None



class filename_from_fullfile(KPF_Primitive):
    """
    This primitive determines the file name from a full path.

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains
                  positional arguments and  keyword arguments passed by the
                  `file_from_kpffile` event issued in the recipe:

            - `action.args[0] (string)`: filename in kpf format
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        success = len(self.action.args) == 1 and isinstance(self.action.args[0], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        fullpath = self.action.args[0]  # e.g. '/data/L0/20230724/KP.20230720.12345.67.fits'
        filename = fullpath.rsplit('/', 1)[-1]  # e.g., 'KP.20230720.12345.67.fits'

        return Arguments(filename)

def print_shell_script_docstring(script_path):
    """
    Extracts and prints the docstring of a shell script.
    """
    try:
        with open(script_path, "r") as file:
            script_lines = file.readlines()

        # Define the start and end markers for the docstring
        docstring_start = "read -r -d '' SCRIPT_DOC <<'EOF'"
        docstring_end = "EOF"

        # Initialize variables
        in_docstring = False
        docstring = []

        # Parse lines to find the docstring
        for line in script_lines:
            if docstring_start in line:
                in_docstring = True
                continue
            if docstring_end in line:
                in_docstring = False
                continue
            if in_docstring:
                docstring.append(line.rstrip())

        # Print the docstring
        if docstring:
            print("\n".join(docstring))
        else:
            print("No docstring found in the script.")

    except FileNotFoundError:
        print(f"Error: File not found at {script_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


class text_file_line_count(KPF_Primitive):
    """
    This primitive determines the number of lines in a text file,
    and returns the number as an integer.

    Description:
        - `action (keckdrpframework.models.action.Action)`: `action.args` contains
                  positional arguments and  keyword arguments passed by the
                  `file_from_kpffile` event issued in the recipe:

            - `action.args[0] (string)`: filename in kpf format
    """

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        KPF_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        success = len(self.action.args) == 1 and isinstance(self.action.args[0], str)
        return success

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        filename = self.action.args[0]  # e.g. '/data/masters/20250101/kpf_20250101_master_flat_GREEN_CCD.csv'
        with open(filename, "r") as file:
            lines = file.readlines()
            num_lines = len(lines)

        return Arguments(num_lines)
