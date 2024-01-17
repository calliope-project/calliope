# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calliope logging examples
#
# In this notebook, we will look at ways of capturing calliope logging outputs and printing them to the console or to file.

# %%
import logging
from pathlib import Path

import calliope

# %% [markdown]
# ## Using internal Calliope functionality
# The `calliope.set_log_verbosity` method allows you to quickly set the logging level across all calliope loggers.
# It doesn't require you to know anything about the `logging` package, just the available [logging levels](https://docs.python.org/3/library/logging.html#logging-levels).

# %% [markdown]
# This is the default log verbosity that is set on importing calliope.
# It will print the WARNING and INFO log levels to the console and it will print the solver output (which is otherwise at the DEBUG log level)

# %%
calliope.set_log_verbosity("info")
m = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-01"])
m.build()
m.solve()

# %% [markdown]
# This will print WARNING and INFO log levels to the console but *NOT* the log solver output

# %%
calliope.set_log_verbosity("info", include_solver_output=False)
m = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-01"])
m.build()
m.solve()

# %% [markdown]
# You can set the log verbosity to print all DEBUG level logs to the console

# %%
calliope.set_log_verbosity("debug")
m = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-01"])
m.build()
m.solve()

# %% [markdown]
# ## Adding your own console logging handler
# If the `calliope.set_log_verbosity` method isn't providing you with enough flexibility then you can add your own logging `handlers`

# %%
# Grab the calliope logger, which will also automatically all the child loggers (e.g., `calliope.core.model`).
logger = logging.getLogger("calliope")

# Remove existing handlers (i.e., those introduced by `calliope.set_log_verbosity` above)
logger.handlers.clear()

# You can define your own custom formatter.
# See https://docs.python.org/3/library/logging.html#logrecord-attributes for available attributes.
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add a ConsoleHandler (this is what `calliope.set_log_verbosity` is doing under the hood)
console_handler = logging.StreamHandler()
console_handler.setLevel(
    logging.INFO
)  # In this example, we only want to see warnings in the console
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# You can also use logging in your scripts to add more information
logger.info("Loading the national-scale example model")
m = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-01"])

logger.info("Building the national-scale example model optimisation problem")
m.build()

logger.info("Solving the national-scale example model optimisation problem")
m.solve()

# %% [markdown]
# ## Adding your own file logging handler
# You may find it more practical to store logging information in files, particularly if you are running your model on a remote device or if you have a *very* large model.
#
# Then, you can search through your logs using your favourite IDE.

# %%
log_filepath = Path(".") / "outputs" / "5_logging"
log_filepath.mkdir(parents=True, exist_ok=True)

# %%
# Grab the calliope logger, which will also automatically all the child loggers (e.g., `calliope.core.model`).
logger = logging.getLogger("calliope")

# Remove existing handlers (i.e., those introduced earlier in this notebook)
logger.handlers.clear()

# You can define your own custom formatter.
# See https://docs.python.org/3/library/logging.html#logrecord-attributes for available attributes.
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Set up a file handler, which will store log outputs in a file
file_handler = logging.FileHandler(log_filepath / "calliope.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# You can also use logging in your scripts to add more information
logger.info("Loading the national-scale example model")
m = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-01"])

logger.info("Building the national-scale example model optimisation problem")
m.build()

logger.info("Solving the national-scale example model optimisation problem")
m.solve()

# %% [markdown]
# Notice that there is no logging to the console here, but that there is now a file `outputs/calliope.log` that contains the logging information.
#
# We can also log both to the console at one level and to file at another:

# %%
# Grab the calliope logger, which will also automatically all the child loggers (e.g., `calliope.core.model`).
logger = logging.getLogger("calliope")

# Remove existing handlers (i.e., those introduced earlier in this notebook)
logger.handlers.clear()

# You can define your own custom formatter.
# See https://docs.python.org/3/library/logging.html#logrecord-attributes for available attributes.
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add a ConsoleHandler (this is what `calliope.set_log_verbosity` is doing under the hood)
console_handler = logging.StreamHandler()
# Log to console at the INFO level
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Set up a file handler, which will store log outputs in a file
file_handler = logging.FileHandler(log_filepath / "calliope.log")
# Log to file at the DEBUG level
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# You can also use logging in your scripts to add more information
logger.info("Loading the national-scale example model")
m = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-01"])

logger.info("Building the national-scale example model optimisation problem")
m.build()

logger.info("Solving the national-scale example model optimisation problem")
m.solve()

# %% [markdown]
# The log file will contain all calliope child logger outputs in one place.
# You will notice the name of the logger, which corresponds to the file where the log was recorded, at the second level of the log messages.
#
# We can store each of these child loggers to a different file if we like:

# %%
# Grab the calliope logger, which will also automatically all the child loggers (e.g., `calliope.core.model`).
logger = logging.getLogger("calliope")

# Remove existing handlers (i.e., those introduced earlier in this notebook)
logger.handlers.clear()

# You can define your own custom formatter.
# Here we don't include the logger name, as the filename will contain that information.
# See https://docs.python.org/3/library/logging.html#logrecord-attributes for available attributes.
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

for logger_name in logging.root.manager.loggerDict.keys():
    if not logger_name.startswith("calliope"):
        # There are lots of other loggers that calliope imports from its dependencies which we will ignore.
        # You can also dump these log to files if you like, by removing this conditional statement.
        continue
    # Set up a file handler, which will store log outputs in a file
    file_handler = logging.FileHandler(log_filepath / f"{logger_name}.log")
    # Log to file at the DEBUG level
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.getLogger(logger_name).addHandler(file_handler)

m = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-01"])
m.build()
m.solve()
