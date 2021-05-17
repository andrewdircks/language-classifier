DEV_DB_PATH = "snippets-dev/snippets-dev.db"
FULL_DB_PATH = None

class Method:
    CHAR_COUNT = 1
    N_GRAMS = 2
    TEXT_VECTORIZATION = 3

langs = [
    "Bash",
    "C",
    "C++",
    "CSV",
    "DOTFILE",
    "Go",
    "HTML",
    "JSON",
    "Java",
    "JavaScript",
    "Jupyter",
    "Markdown",
    "PowerShell",
    "Python",
    "Ruby",
    "Rust",
    "Shell",
    "TSV",
    "Text",
    "UNKNOWN",
    "YAML"
]

langs_ace = [
    "sh",
    "c_cpp",
    "c_cpp",
    "a",
    "dot",
    "golang",
    "html",
    "json",
    "java",
    "javascript",
    "plain_text", # jupyter is nothing
    "markdown",
    "powershell",
    "python",
    "ruby",
    "rust",
    "sh",
    "plain_text", # tab-seperated values
    "plain_text",
    "plain_text", # unknown
    "yaml"

]

chars = "\t\nabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%^&*~`+-=<>()[]{}\" "

snippet_len = 256

def _dictify(seq):
    return {s: i for i, s in enumerate(seq)}

langs_map = _dictify(langs)
n_langs = len(langs)
chars_map = _dictify(chars)
n_chars = len(chars)

BATCH_SIZE = 1000
NUM_EPOCHS = 20

N_GRAMS_PADDING = 100
N_GRAMS = 2

MAX_FEATURES = 10000
SNIPPET_LENGTH = 250

PERCENT_TEST = 0.2