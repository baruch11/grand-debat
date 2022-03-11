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


def display_questions_from_json(filepath):
    """Display ids and questions from json file.

    Args:
        filepath (str): path of the json
    """
    f = open(filepath, "r")
    objects = ijson.items(f, 'item')

    for answ in objects:
        questions_id = [r['questionId'] for r in answ['responses']]
        questions_title = [r['questionTitle'] for r in answ['responses']]
        break  # only one entry in the json is needed
    for idq, title in zip(questions_id, questions_title):
        print("Qestion id : {}\n{}\n".format(idq, title))


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
