from predic import predict_db
from match import match
from typing import List, Tuple, Callable, Any
def chatbot(predic: Tuple[int,float]):
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


    # The projection functions, that give us access to certain parts of a "movie" (a tuple)
    def get_month(movie: Tuple[str, str, int, List[str]]) -> str:
        return movie[0]

    def get_price(movie: Tuple[str, str, int, List[str]]) -> int:
        return movie[2]

    # Below are a set of actions. Each takes a list argument and returns a list of answers
    # according to the action and the argument. It is important that each function returns a
    # list of the answer(s) and not just the answer itself.


    def price_by_month(matches: List[str]) -> List[str]:
        """Finds price in the passed in month

        Args:
            matches - a list of 1 string, just the month. Note that this month is passed as a
                string and should be converted to an int

        Returns:
            a list of movie titles made in the passed in year
        """
        #converting string to int month 

        # done converting

        for predic in predict_db:
            if int(matches[0]) == get_month(predic):
                
                return (get_price(predic))
        
        
    # dummy argument is ignored and doesn't matter
    def bye_action(dummy: List[str]) -> None:
        raise KeyboardInterrupt


    # The pattern-action list for the natural language query system A list of tuples of
    # pattern and action It must be declared here, after all of the function definitions
    pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [
        (str.split("what will be the price of corn in _"), price_by_month),
        (str.split("what movies were made between _ and _"), ),
        (str.split("what movies were made before _"), title_before_year),
        (str.split("what movies were made after _"), title_after_year),
        # note there are two valid patterns here two different ways to ask for the director
        # of a movie
        (str.split("what will be the price of corn in %"), price_by_month),
        (str.split("what will the price be on %"), price_by_month),
        (str.split("what will the corn price be on %"), price_by_month),
        (str.split("what will the corn price be in %"), price_by_month),
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
        time=match(pat,src)
        x=0
        while(x<time.len())
            
        
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

        print("\nThankyou for your time!\n")


    # uncomment the following line once you've written all of your code and are ready to try
    # it out. Before running the following line, you should make sure that your code passes
    # the existing asserts.
    query_loop()

# method calls
main()