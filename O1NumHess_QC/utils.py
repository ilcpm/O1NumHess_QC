import importlib.util
import os
import shutil
from pathlib import Path
from typing import Dict, Union

__all__ = ["getConfig", "init_config", "getAbsPath", "_PROGRAMS"]

def getAbsPath(path: Union[str, Path]) -> Path:
    """Parse path, expand ``~``, make absolute, return normpath."""
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(path))))

_PROGRAMS = ["BDF", "ORCA"]
_template_config_dir = getAbsPath(Path(__file__).parent / "template")
_system_config_dir = getAbsPath(Path(__file__).parent / "config")
_user_config_dir = getAbsPath(Path("~/.O1NumHess_QC"))

def getConfig(program: str, config_name: str = "") -> Dict[str, str]:
    """Load one config entry from the user config module.

    The config file is divided into system config file and user config file. The system config file is located in the package installation directory, while the user config file is located in the ``.O1NumHess_QC`` directory in the user's home directory. The program will first try to load the system config file (in this case, the user config file is ignored), if the system config file does not exist, then it will try to load the user config file. If both config files do not exist, an exception will be raised.

    **Note:** The system config file is reserved for special cases, and is generally not recommended for use. The user config file is the one that general users should use. Users can initialize the user config file (or the system config file, but not recommended) by calling the ``O1NumHess_QC.init_config()`` function. After initialization, users need to edit the config file to add their own config, and the format can be directly referred to the generated template file.

    Args:
        program (str): Program name, such as ``"BDF"`` or ``"ORCA"``.
        config_name (str): Optional entry name. If empty, the first entry is used.

    Returns:
        config (Dict[str, str]): A config dictionary with runtime shell environment and executable path.

    Raises:
        ValueError: If the program is not supportted.
        FileNotFoundError: If the config file does not exist.
        ImportError: If the config module cannot be loaded.
        AttributeError: If the config structure is invalid or the entry is not found.
    """
    if program not in _PROGRAMS:
        raise ValueError(f"program {program} is not supported!")

    # try to load system config file
    system_config_file = _system_config_dir / f"{program}_config.py"
    # load user config file if system config file does not exist
    user_config_file = _user_config_dir / f"{program}_config.py"

    file = None
    if system_config_file.exists():
        file = system_config_file
        # print(f"Using system config file: {file}")
    elif user_config_file.exists():
        file = user_config_file
        # print(f"Using user config file: {file}")
    else:
        raise FileNotFoundError(f"the config file of {program}: system config and {user_config_file} do not exist, refer to the document")
    module_name = file.stem # "{program}_config.py"
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None:
        raise ImportError(f"something wrong while importing the config file {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore (ignore warning from Pylance)

    # print(module.config)
    try:
        # if config_name is empty, use the first entry
        if config_name == "":
            config = module.config[0]
        else:
            for dic in module.config:
                if dic["name"] == config_name:
                    config = dic
    except IndexError:
        raise AttributeError(f"the config file {file} is empty")
    except AttributeError:
        raise AttributeError(f"something wrong with the config file {file}")
    try:
        return config  # type: ignore
    except NameError:
        raise AttributeError(f"the config file {file} does not have the config name: '{config_name}'")

def check_config_exist():
    """check if any config exists"""
    for program in _PROGRAMS:
        try:
            getConfig(program)
            return True
        except FileNotFoundError:
            continue
    return False

def init_config():
    """Initialize config by creating system or user config directory and copying config template into it

    The config file is divided into system config file and user config file. The system config file is located in the package installation directory, while the user config file is located in the ``.O1NumHess_QC`` directory in the user's home directory. The program will first try to load the system config file (in this case, the user config file is ignored), if the system config file does not exist, then it will try to load the user config file. If both config files do not exist, an exception will be raised.

    The system config file is reserved for special cases, and is generally not recommended for use. The user config file is the one that general users should use. Users can initialize the user config file (or the system config file, but not recommended) by calling the ``O1NumHess_QC.init_config()`` function. After initialization, users need to edit the config file to add their own config, and the format can be directly referred to the generated template file.
    """

    # ask user if they want to init config in system config directory or user config directory
    while True:
        choice = input(
            "Where do you want to initialize config in?\n" +\
            "\t1) user config directory (the directory in your home directory, **recommanded**)? \n" +\
            "\t2) system config directory (the directory where the package is installed, **not recommended**)\n" +\
            "\t\tsystem config is for special design, not suggested for general users\n"
            "([u] for user/[s] for system):").lower()
        if choice in ["u", "s"]:
            break
        print()
        print("invalid choice, please enter 'u' for system or 's' for user")
    # system config directory is the directory where the package is installed, user config directory is the directory in user's home directory
    if choice == "s":
        config_dir = _system_config_dir
    else:
        config_dir = _user_config_dir
    if not config_dir.exists():
        config_dir.mkdir(parents=True)

    new_config = []
    for program in _PROGRAMS:
        print("*" * 60)
        template_config_file = _template_config_dir / f"{program}_config.py"
        if not template_config_file.exists():
            print(f"template config file for {program} does not exist, skipping initialization")
            continue
        config_file = config_dir / f"{program}_config.py"

        # if the config file already exists, ask user if they want to overwrite it
        if config_file.exists():
            while True:
                overwrite = input(f"the **{program}** config file \n\t{config_file}\n already exists, do you want to overwrite it? (y/n):").lower()
                if overwrite in ["y", "n"]:
                    break
                print()
                print("invalid choice, please enter 'y' for yes or 'n' for no")
            if overwrite != "y":
                print(f"skipping {program} config initialization")
                continue
        # copy and paste file into user config directory
        shutil.copy(template_config_file, config_file)
        new_config.append(program)
        print(f"config file for {program} has been initialized at {config_file}")

    print("=" * 60)
    for program in new_config:
        config_file = config_dir / f"{program}_config.py"
        if config_file.exists():
            print(f"config file for {program} is located at \n\t{config_file}")
        else:
            print(f"config file for {program} does not exist, refer to the document")
    print("now **you should edit these files to add your config**! see the contents in these files for details")

if __name__ == "__main__":
    # test
    print(getConfig("BDF"))
    # print(getConfig("BDF", "BDf")) # error test
    # print(getAbsPath("/mnt/~/abc/dfs"))
    # print(getAbsPath("~/abc/dfs"))
    # print(getAbsPath("../abc/dfs"))
