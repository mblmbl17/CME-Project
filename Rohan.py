from Neuralnet import main
from project import model

# Important variables:
#     movie_db: list of 4-tuples (imported from movies.py)
#     pa_list: list of pattern-action pairs (queries)
#       pattern - strings with % and _ (not consecutive)
#       action  - return list of strings

# THINGS TO ASK THE MOVIE CHAT BOT:
# what movies were made in _ (must be date, because we don't have location)
# what movies were made between _ and _
# what movies were made before _
# what movies were made after _
# who directed %
# who was the director of %
# what movies were directed by %
# who acted in %
# when was % made
# in what movies did % appear
# bye

#  Include the movie database, named movie_db
from movies import movie_db
from match import match
from typing import List, Tuple, Callable, Any

# The projection functions, that give us access to certain parts of a "movie" (a tuple)
def get_title(movie: Tuple[str, str, int, List[str]]) -> str:
    return movie[0]


def get_director(movie: Tuple[str, str, int, List[str]]) -> str:
    return movie[1]


def get_year(movie: Tuple[str, str, int, List[str]]) -> int:
    return movie[2]


def get_actors(movie: Tuple[str, str, int, List[str]]) -> List[str]:
    return movie[3]


# Below are a set of actions. Each takes a list argument and returns a list of answers
# according to the action and the argument. It is important that each function returns a
# list of the answer(s) and not just the answer itself.


def price_by_month(matches: List[str]) -> List[str]:
    """Finds all movies made in the passed in year

    Args:
        matches - a list of 1 string, just the year. Note that this year is passed as a
            string and should be converted to an int

    Returns:
        a list of movie titles made in the passed in year
    """
    results=[]

    for movie in movie_db:
        if int(matches[0]) == get_year(movie):
            
            results.append(get_title(movie))
    
    return results
    


def month_by_price(matches: List[str]) -> List[str]:
    """Finds all movies made in the passed in year range

    Args:
        matches - a list of 2 strings, the year beginning the range and the year ending
            the range. For example, to get movies from 1991-1994 matches would look like
            this - ["1991", "1994"] Note that these years are passed as strings and
            should be converted to ints.

    Returns:
        a list of movie titles made during those years, inclusive (meaning if you pass
        in ["1991", "1994"] you will get movies made in 1991, 1992, 1993 & 1994)
    """
    results=[]
    for movie in movie_db:
        if int(matches[0])<=get_year(movie) and get_year(movie)<= int(matches[1]):
            results.append(get_title(movie))
    return results
    
# dummy argument is ignored and doesn't matter
def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt


# The pattern-action list for the natural language query system A list of tuples of
# pattern and action It must be declared here, after all of the function definitions
pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [
    (str.split("what movies were made in _"), title_by_year),
    (str.split("what movies were made between _ and _"), title_by_year_range),
    (str.split("what movies were made before _"), title_before_year),
    (str.split("what movies were made after _"), title_after_year),
    # note there are two valid patterns here two different ways to ask for the director
    # of a movie
    (str.split("who directed %"), director_by_title),
    (str.split("who was the director of %"), director_by_title),
    (str.split("what movies were directed by %"), title_by_director),
    (str.split("who acted in %"), actors_by_title),
    (str.split("when was % made"), year_by_title),
    (str.split("in what movies did % appear"), title_by_actor),
    (str.split("what movies has % acted in"), title_by_actor),
    (["bye"], bye_action),
]


def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    for pat, act in pa_list:
        mat=match(pat,src)
        # print(pat)
        # print(src)
        # print(act)
        if mat is not None:
            answer=act(mat)
            print(answer)
            return answer if answer else ["No answers"]
    return ["I don't understand"]    


def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully.
    """
    print("Welcome to the movie database!\n")
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nSo long!\n")


# uncomment the following line once you've written all of your code and are ready to try
# it out. Before running the following line, you should make sure that your code passes
# the existing asserts.
# query_loop()

if __name__ == "__main__":
    assert isinstance(title_by_year(["1974"]), list), "title_by_year not returning a list"
    assert isinstance(title_by_year_range(["1970", "1972"]), list), "title_by_year_range not returning a list"
    assert isinstance(title_before_year(["1950"]), list), "title_before_year not returning a list"
    assert isinstance(title_after_year(["1990"]), list), "title_after_year not returning a list"
    assert isinstance(director_by_title(["jaws"]), list), "director_by_title not returning a list"
    assert isinstance(title_by_director(["steven spielberg"]), list), "title_by_director not returning a list"
    assert isinstance(actors_by_title(["jaws"]), list), "actors_by_title not returning a list"
    assert isinstance(year_by_title(["jaws"]), list), "year_by_title not returning a list"
    assert isinstance(title_by_actor(["orson welles"]), list), "title_by_actor not returning a list"
    assert isinstance(title_by_director(["jonathan demme"]), list), "title_by_director not returning a list"

    assert sorted(title_by_year(["1974"])) == sorted(
        ["amarcord", "chinatown"]
    ), "failed title_by_year test"
    assert sorted(title_by_year_range(["1970", "1972"])) == sorted(
        ["the godfather", "johnny got his gun"]
    ), "failed title_by_year_range test"
    assert sorted(title_before_year(["1950"])) == sorted(
        ["casablanca", "citizen kane", "gone with the wind", "metropolis"]
    ), "failed title_before_year test"
    assert sorted(title_after_year(["1990"])) == sorted(
        ["boyz n the hood", "dead again", "the crying game", "flirting", "malcolm x"]
    ), "failed title_after_year test"
    assert sorted(director_by_title(["jaws"])) == sorted(
        ["steven spielberg"]
    ), "failed director_by_title test"
    assert sorted(title_by_director(["steven spielberg"])) == sorted(
        ["jaws"]
    ), "failed title_by_director test"
    assert sorted(actors_by_title(["jaws"])) == sorted(
        [
            "roy scheider",
            "robert shaw",
            "richard dreyfuss",
            "lorraine gary",
            "murray hamilton",
        ]
    ), "failed actors_by_title test"
    assert sorted(actors_by_title(["movie not in database"])) == [], "failed actors_by_title not in database test"
    assert sorted(year_by_title(["jaws"])) == sorted(
        [1975]
    ), "failed year_by_title test"
    assert sorted(title_by_actor(["orson welles"])) == sorted(
        ["citizen kane", "othello"]
    ), "failed title_by_actor test"
    assert sorted(search_pa_list(["hi", "there"])) == sorted(
        ["I don't understand"]
    ), "failed search_pa_list test 1"
    assert sorted(search_pa_list(["who", "directed", "jaws"])) == sorted(
        ["steven spielberg"]
    ), "failed search_pa_list test 2"
    assert sorted(
        search_pa_list(["what", "movies", "were", "made", "in", "2020"])
    ) == sorted(["No answers"]), "failed search_pa_list test 3"
    assert sorted(title_by_director(["jonathan demme"])) == sorted(
        ["silence of the lambs"]
    ), "failed title_by_director test"

    print("All tests passed!")


# calls the regression model 
model()

main()