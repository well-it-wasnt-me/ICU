Installation & Configuration
============================

.. _installation:

Installation
------------

Now, installing this script is easier than forgetting your ex's birthday.

But just in case you're the type who needs a roadmap to get out of a paper bag, here's how you do it:

==================
Repository Cloning
==================
First things first, let's get this show on the road. Open up your terminal...you know, that black box where magic happens...and run:

.. code-block:: console

    $ git clone https://github.com/well-it-wasnt-me/ICU.git
    $ cd ICU

===================================
Set Up a Python Virtual Environment
===================================
Because isolating problems is the first step to solving them, or so my therapist says, we're gonna set up a virtual environment.

Think of it as a playpen for your Python packages.

.. code-block:: console

    $ python3 -m venv venv
    $ source venv/bin/activate
    (venv) $ pip install -r requirements.txt

.. _configuration:

Configuration
----------------

Before you run the script, you need to set up the `cameras.yaml` file.

Don't worry...it's easier than assembling IKEA furniture.

Just rename/copy `cameras-example.yaml` to `cameras.yaml` and fill in the blank !
