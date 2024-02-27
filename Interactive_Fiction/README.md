
# CIS7000 - Interactive Fiction - Homework 2

## Part 1 Reflection on Hallucinations and Game State

### Playthrough - 1

1. "pick rose"
    * Always hallucinates smelling the rose even without the explicit command

2. "catch fish"
    * Hallucinates using the pole to catch the fish even without stating that the pole is used

3. Directions are conflated.
    * "South" and "down" are used interchangeably as are "North" and "up". The model only seems to confuse these when using
      them in evocative descriptions, but not in the actual parsing of directions.

4. Feeding fish to the troll triggers movement across the drawbridge.
    * The game requires that you cross the drawbridge after feeding the troll but GPT hallucinates that you run across all at once.
      * The game state isn't affected, however, and the player was able to cross the drawbridge at the next move.

5. Hallucinates that there are torches on the wall of the courtyard
    * This in interesting because someone playing the game could think that these are gettable items, making this a
      particularly problematic response.

6. Hallucinates picking up the key and sword after hitting the guard
    * Suggests that the player picks up both items even though this command hasn't been given.
    * Neither item is found in the inventory as expected.
    * The player is able to pick up the key at the next turn.

7. Hallucinations in the feasting hall
    * Long wooden table
    * Tapestries on the walls
    * Statues leading to the throne room
    * After the player put the crown on their head and returned to the feasting hall, GPT hallucinated: Roasted meats and spiced wine.

8. Hallucinates that the lamp is lit 
    * Creates a box of matches that are used to light the lamp
    * As with the torches, these could be perceived as gettable items, so this is especially problematic.

9. Hallucinates moss and water in the dungeon
    * Again, these could be perceived as gettable items, which is troublesome.

11. Placing the crown on your head
    * This triggered the hallucination that the player was the ruler of the kingdom, which should technically win the game.
      However, this condition is not met until the player sits upon the throne.
   
### Playthrough - 2

1. Cottage:
    * fireplace
    * cobwebs: tried to get but couldn’t
2. Garden path:
    * cobblestones
    * moss
3. fishing pond:
    * pier to the east
    * frog
    * tackle box on the pier
    * Use pole to fish when not specified
    * failed return still describes a successful movement from pond to garden
4. Drawbridge
    * the troll discards fish bones, but they are not gettable when I try to pick them up
5. Courtyard:
    *  key in possession but also on ground when returning to courtyard
6. Tower stairs:
    * Somehow teleported from courtyard to stairs with “go to feasting hall”
7. Feasting hall:
    * failed command hallucinates (in the style of the original game)
         You are in the Great Feasting Hall.
         Exits:
         West to Courtyard
         East to Throne Room
         
         You see:
          * a long, grand table
          * a chandelier
          * a fireplace
                 look in fireplace
                 get chandelier
                 get table
8. Dungeon stairs:
    * lamp in inventory but light command fails (GPT says no inventory is there)
    * try to relight and the lamp is already lit
9. Dungeon
    * “light candle with lamp” fails and says the candle is already lit
  
### Playthrough - 3

There were more hallucinations than I outlined below, but the main purpose of this
additional run was to try to interact with hallucinated objects to see what happened.
This run also occured after fixing the limit_context_length() function to prevent the
error that occured at the end of game_transcript_with_GPT_descriptions-2024-02-19_22-15(early_run).

1. Tower:
    * It said that the tower had a small desk and a bed
    * When I tried to sit at the desk it teleported me back down to the courtyard
3. Feasting hall:
    * It said that there were meats and wine on the table
    * When I tried to eat the meat, it said I was attempting to "interact with
      an object that is not interactive in the game."
    * When I tried to drink the wine, it crashed throwing the following error:
      "TypeError: get_items_in_scope() got an unexpected keyword argument 'hint'"


## Part 2 Limitations of intent determination

1. Reliance on hints for determining which item to act upon
    * This is especially important for the custom actions, which were failing without the addition of hints in the action
      classes for Read_Runes, Wear_Crown, Sit_On_Throne, and Unlock_Door. We added hints to these in the
      action_castle.ipynb file to fix this issue. Afterwards, all of these commands worked.

2. Testing intent of commands:
    * We extented the parser.determine_intent() cases to include additional default commands (especially performed by NPC's),
      as well as custom commands like propose, sit on throne, read runes, wear crown, and unlock door.
      * These helped us tweak the prompting for our determine_intent() method until we got reliable results. Originally,
        we were running into an issue where the "grab fishing pole" command triggered the "catch fish" action.
      * We told GPT to pay special attention to the verbs in the command, and we included some examples of similar verbs
        to the main action verb in several commands.
      * The model untimately went 61/61 on these test cases
         * see hw2.ipynb for all of these cases
    * We created test cases with a sample parser and item dictionaries.
      * Set up a variety of non-standard commands and passed these to the parser.get_character(), parser.match_item(),
         and parser.get_direction() commands.
      * for parser.match_item() to work well, we needed to pass in the command as well as a list of items in the inventory/location
         * Before doing this, for the "pick rose" command, parser.match_item() would confuse the hint="rosebush" item as
           "rose" whenever a rose was already in the player's inventory (this test scenario could never occur in the real game).
           After including a list of all items in the location and inventory in the prompt, this problem went away
           (parser.match_item() needed to see that a rosebush was present at the scene).
      * parser.get_character() went 49/49
         * see hw2.ipynb for all of these cases
      * parser.match_item() went 36/36
         * see hw2.ipynb for all of these cases
      * parser.get_direction() went 10/10 (this method was tested more via playing the game a few times)
         * see hw2.ipynb for all of these cases

4. Difficult actions - All custom actions
    * We had to revise how the parser setup occurred
    * The initial method only added commands and their descriptions upon the initialization of the GptParser2() module.
    * This causes problems because the custom actions are only added to the game upon calling `game.set_parser()`.
    * By wrapping the original `__init__()` code in a method, `refresh_command_list()`, we could add the custom actions
      to the list of command descriptions passed to GPT.
    * This allowed it to properly identify the custom actions when they were invoked in the game.



## Part 3 Reflection on Troubesome Commands

1. "grab the rod"
    * Early on, this command regularly gave us trouble. The parser often confused the initiator of the action to be the troll instead of the player. We emphasized toward the end of the prompt that it should return "The player" if it was unsure of who was being described by the action hint. This fixed the problem.

2. Desciption of items and exists available
    * We found that GPT struggled to describe the exits and the items that were present at every location. It would occasionally
      hallucinate the direction to a location. For example, describing that the garden path was located to the North of the cottage,
      instead of simply outside of the cottage.
    * When up in the tree, it also thought that the Winding Path was South (instead of Down) and that the Drawbridge was to the East,
      even though it's to the East of the Winding Path below.

3. "go fishing"
    * This command is difficult because the current implementation of the fishing action does not allow for GPT interpretation.
      The phrase, "with pole" is hardcoded into the game and the command will succeed only if this string is present in the command.
      We tried many variations of the command that GPT clearly understood as an act of fishing, but these commands failed unless
      the phrase "with pole" was included. 
    * Alternative commands:
      * "cast line to catch a fish"
      * "use rod to fish"
    * Despite failing the command, the output description often included the player using a pole to try to catch a fish.
      I actually like this behavior because it suggests that the model understands the intent of fishing and that the presence
      of the fishing pole in the player's inventory means they ought to use it for fishing. It is simply a quirk of the game
      that the action fails.
       * Interestingly, it tried to utilize the passed game response about the fish being too fast for the player's hand,
           but in the context of using the fishing pole. To paraphrase, it would say something like, your hook sinks into
           the water and a fish bites, but as you try to reel it in the fish is too fast for your hands.

4. "Propose marriage to princess"
    * Initially, this failed because there were no hints for the "proposer" and "propositioned" characters. After adding these,
      the propose action succeed and was surprisingly robust. In action_castle.ipynb, the Propose class utilizes a
      split_command() method on the word "propose", getting the pieces of the command before and after this word –
      this helps identify the "proposer" and "propositioned". However, even when we didn't include the word "propose",
      as in "ask for the princess's hand in marriage", it still worked. Using just the hints of "proposer" and
      "propositioned" were enough for GPT to correctly identify both characters involved.

5. "Return to the fallen guard"
    * While GPT was able to consistently identify directions by specifying the adjacent location we wanted the player to visit
      (e.g. saying "head the fishing pond" from the garden path), we tried to see if it could identify the desired location based
      on a character there – "return to the fallen guard." However, it was unable to connect the location of the fallen guard
      in the command history with the desired movement from the feasting hall to the courtyard. I think that something like
      "head the courtyard" works because "courtyard" is in the list of exits. On the other hand, the extra step of checking the command history
      to find "fallen guard" in the "courtyard", and then seeing that "courtyard" appears as an exit to the West was a bit too much.


