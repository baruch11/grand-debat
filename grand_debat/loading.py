"""Loading datasets functions"""
import ijson
from tqdm import tqdm


def load_answers(filepath, selected_question):
    """Load answers from json file filepath.

    Args:
        filepath (str): input json file
        selected_question (str) : id of the question
    Returns:
        list of str: the answers
    """
    with open(filepath, "r") as f:
        entries = ijson.items(f, 'item')

        ret = []
        for entry in tqdm(entries):
            for quest in entry["responses"]:
                if (quest.get('questionId') == selected_question and
                        quest.get('value') is not None):
                    ret.append(quest.get('value'))
    return ret


def get_questions_from_json(filepath):
    """Get ids and questions from json file.

    Args:
        filepath (str): path of the json
    Returns:
        dict, key = id, values = questions
    """
    with open(filepath, "r", encoding='utf8') as fjson:
        objects = ijson.items(fjson, 'item')

        for answ in objects:
            ret = {r['questionId']:r['questionTitle'] for r in answ['responses']}
            break  # only one entry in the json is needed
        return ret


def get_path(themes, selected_theme):
    """
    Returns the path of the dowloaded file. Useful in case several themes have
    been downloaded

    Parameters
    ----------
    themes: [str]
        List of themes
    selected_theme: int
        Item number

    Returns
    -------
    path: str
        Path to relevant theme data
    """
    for theme, link in themes.items():
        if int(theme[0]) == selected_theme:
            path = './data/' + theme[2:] + '.json'
    return path
