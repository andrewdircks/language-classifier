DEV_DB_PATH = "snippets-dev/snippets-dev.db"
FULL_DB_PATH = None

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