# Text Adventure Games!

This Python notebook builds a simple text advenutre game inspired by the
[Adventuron Classroom](https://adventuron.io/classroom/) design by Chris
Ainsley of Adventuron Software Limited.

The main components are:
1. __The parser__, which interprets the player's commands.
2. __The game__, which represents world (a collection of __locations__,
   __items__, and __characters__), and describes what the player sees.
3. __The data__, which you input to create your own unique game.

For this homework assignment, I have implemented most of a classic text
adventure game for you.  I marked a few places that you'll need to fill in your
own code in order to implement ACTION CASTLE!

You can download a PDF of the game from [Parsely: Preview n' Play Edition (this
contains a free copy the Action Castle game).](http://www.memento-mori.com/pdf/parsely-preview-n-play-edition)


## Environment Setup

First, get the the homework repo, then create a virtual environment, and then install
the dependencies.

There are two ways that you can get the homework repository.  For students, you can [download the zip file list in the homework instructions](https://interactive-fiction-class.org/homeworks/text-adventure-game/text-adventure-game.html).  TAs can use `git clone https://github.com/interactive-fiction-class/interactive-fiction-class-homework`.
```
$ unzip CIS-7000.zip
$ cd CIS-7000/HW1
$ python3 -mvenv venv
$ source venv/bin/activate
(venv) $ pip install .
```

### Graphviz

If you want to use the anything from `text_adventure_games.viz`, you will also need to install graphviz. There is both the graphviz app and the graphviz python library, which is a wrapper for the app.

We'll let the graphviz project explain how to do that: [graphviz.org/download/](https://graphviz.org/download/)

### Dev Environment Setup

If you are part of the dev team for this repo, use this pip command instead,
to do an editable install of this library, and to install the black code
formatter.

```
(venv) $ pip install -e .[dev]
```

## VSCode

We like to use VSCode, but you can use any editor you feel comfortable with.
The best way to ensure a good experience with VSCode is to launch it from the
project directory with your virtual environment already enabled.

```
$ cd path/to/homework/<repo name>/
$ source venv/bin/activate
(venv) $ code .
```

If the `code` command does not work, open VSCode's command pallet (command+shift+p)
and type `shell command`. VSCode should then show an option for installing
'code' in your PATH.


### The Traditional Way

For folks who prefer traditional Jupyter, or perhaps Jupyter Lab, launch the
Jupyter server from the homeworks directory and everything should work as you
expect.

```
$ cd path/to/homework/<repo name>/
$ source venv/bin/activate
$ cd homeworks/
$ jupyter notebook
```

