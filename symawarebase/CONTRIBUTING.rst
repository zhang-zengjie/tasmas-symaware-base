============
Contributing
============

Welcome to |project-name| contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but `other kinds of contributions`_ are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at `contribution-guide.org`_. Other resources are also
listed in the excellent `guide created by FreeCodeCamp`_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, `Python Software
Foundation's Code of Conduct`_ is a good reference in terms of behavior
guidelines.


Issue Reports
=============

If you experience bugs or general issues with |project-name|, please have a look
on the `issue tracker`_. If you don't see anything useful there, please feel
free to fire an issue report.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


Documentation Improvements
==========================

You can help improve |project-name| docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

|project-name| documentation uses Sphinx_ as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.

Most of the documentation is written in reStructuredText_ and can be found in
the ``docs`` folder. The main documentation page is ``docs/index.rst``.
Some markdown files are also in use.

Finally, to document the code, |project-name| uses `docstrings`_ in the
`Numpy style`_.
They can be found in the source code files at the beginning of functions, 
classes and modules.


.. tip::
    Please notice that the GitLab web interface provides a quick way of
    propose changes in |project-name|'s files. While this mechanism can
    be tricky for normal code contributions, it works perfectly fine for
    contributing to the docs, and can be quite handy.

    If you are interested in trying this method out, please navigate to
    the ``docs`` folder in the source repository_, find which file you
    would like to propose changes and click in the edit button at the
    top right, to open `GitLab's code editor`_ or  `GitLab's IDE`. Once you finish editing the file,
    please write a message in the form at the bottom of the page describing
    which changes have you made and what are the motivations behind them and
    submit your proposal.

When working on documentation changes in your local machine, you can
compile them using |tox|_::

    tox -e docs

and use Python's built-in web server for a preview in your web browser
(``http://localhost:8000``)::

    python3 -m http.server --directory 'docs/_build/html'

Before submitting your changes, check for potential errors by running::

    tox -e doctests,linkcheck


Code Contributions
==================

|project-name| is a Python package.
It can be installed with ``pip install symaware-base``.

It is part of the larger ``symaware`` project, which is a collection of tools
aimed at addressing the fundamental need for a new conceptual framework for awareness
in multi-agent systems (MASs) that is compatible with the internal models
and specifications of robotic agents and that enables safe simultaneous operation of collaborating autonomous agents and humans.

The SymAware framework will use compositional logic, symbolic computations, formal reasoning, and uncertainty quantification
to characterise and support situational awareness of MAS in its various dimensions,
sustaining awareness by learning in social contexts, quantifying risks based on limited knowledge,
and formulating risk-aware negotiation of task distributions.

Directory structure
-------------------

The source code of |project-name| is located in the ``src`` folder.
The ``tests`` folder contains the package's tests and the ``docs`` folder contains
the documentation sources.

.. code-block:: text

    symaware-base
    ├── .pylintrc                # pylint configuration
    ├── docs
    │   ├── _static              # Static files for the documentation
    │   ├── conf.py              # Sphinx configuration
    │   ├── index.rst            # Main page of the documentation
    │   └── requirements.txt     # Documentation requirements
    ├── pyproject.toml           # Project configuration
    ├── README.md
    │── src                      # Source code
    │   └── testsymaware         # Namespace
    │       └── base              # Package
    │           ├── __init__.py  # Source code
    │           └── ...
    ├── tests                    # Package tests
    │   ├── conftest.py          # Optional pytest configuration
    │   ├── unit                 # Unit tests
    │   ├── integration          # Integration tests
    │   └── e2e                  # End-to-end tests
    └── tox.ini                  # tox configuration

Submit an issue
---------------

Before you work on any non-trivial code contribution it's best to first create
a report in the `issue tracker`_ to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

Create an environment
---------------------

Before you start coding, we recommend creating an isolated `virtual
environment`_ to avoid any problems with your installed Python packages.
This can easily be done via either `pip-venv`_::

    python3 -m venv .venv
    source .venv/bin/activate # or .venv\Scripts\activate.bat on Windows

or |virtualenv|_::

    virtualenv .venv
    source .venv/bin/activate # or .venv\Scripts\activate.bat on Windows

or Miniconda_::

    conda create -n symaware-base python=3 six virtualenv pytest pytest-cov
    conda activate symaware-base

Clone the repository
--------------------

#. Create an user account on |the repository service| if you do not already have one.
#. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on |the repository service|.
#. Clone this copy to your local disk::

    git clone git@{{ cookiecutter.repository.split('/')[2] }}:<YOUR FORKED REPOSITORY>.git
    cd symaware-base

#. You should run::

    pip install -e .

   to be able to import the package under development in the Python REPL.

Implement your changes
----------------------

#. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the main branch!

#. Start your work on this branch. Don't forget to add docstrings_ to new
   functions, modules and classes, especially if they are part of public APIs.
   Also, try adding unit tests for your new code to make sure it works as
   expected and to avoid regressions in the future.

   If the changes only modify the source code, 
   they will be limited to the ``src`` folder and the ``tests`` folder.

#. Add yourself to the list of contributors in ``pyproject.toml``.

#. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit

   to record your changes in git_.


.. important:: Don't forget to add unit tests and documentation in case your
    contribution adds an additional feature and is not just a bugfix.

    Moreover, writing a `descriptive commit message`_ is highly recommended.
    In case of doubt, you can check the commit history with::

        git log --graph --oneline --abbrev-commit --all

    to look for recurring communication patterns.

#. Please check that your changes don't break any tests or linting rules with::

    tox

   (after having installed |tox|_ with ``pip install tox`` or ``pipx``).

   You can also use |tox|_ to run several other pre-configured tasks in the
   repository. Try ``tox -av`` to see a list of the available checks.

    * If you want to run test or linting checks selectively, use::

        tox -e py-test
        tox -e py-lint

    * If some of the lint checks fail, it may be possible to automatically fix
      them with::

        tox -e fixlint

    * You can perform the same actions without |tox|_ by running::
        
        # Install additional development dependencies
        pip3 install .[lint,test]

        # Lint
        black --check src tests
        pylint src tests
        mypy src tests
        isort --check-only --diff src tests

        # Test
        pytest

        # Try to automatically fix lint problems
        black src tests
        isort src tests

Submit your contribution
------------------------

#. If everything works fine, push your local branch to |the repository service| with::

    git push -u origin my-feature

#. Go to the web page of your fork and click |contribute button|
   to send your changes for review.


Troubleshooting
---------------

The following tips can be used when facing problems to build or test the
package:

#. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   ``.eggs``, as well as the ``*.egg-info`` folders in the ``src`` folder or
   potentially in the root of your project.

#. Sometimes |tox|_ misses out when new dependencies are added, especially to
   ``setup.cfg`` and ``docs/requirements.txt``. If you find any problems with
   missing dependencies when running a command with |tox|_, try to recreate the
   ``tox`` environment using the ``-r`` flag. For example, instead of::

    tox -e docs

   Try running::

    tox -r -e docs

#. Make sure to have a reliable |tox|_ installation that uses the correct
   Python version (e.g., 3.9+). When in doubt you can run::

    tox --version
    # OR
    which tox

   If you have trouble and are seeing weird errors upon running |tox|_, you can
   also try to create a dedicated `virtual environment`_ with a |tox|_ binary
   freshly installed. For example::

    virtualenv .venv
    source .venv/bin/activate
    .venv/bin/pip install tox
    .venv/bin/tox -e all

#. `Pytest can drop you`_ in an interactive session in the case an error occurs.
   In order to do that you need to pass a ``--pdb`` option (for example by
   running ``tox -- -k <NAME OF THE FALLING TEST> --pdb``).
   You can also setup breakpoints manually instead of using the ``--pdb`` option.


Maintainer tasks
================

Releases
--------

If you are part of the group of maintainers and have correct user permissions
on PyPI_, the following steps can be used to release a new version for
|project-name|:

#. Make sure all unit tests are successful.
#. Tag the current commit on the main branch with a release tag, e.g., ``v1.2.3``.
#. Push the new tag to the upstream repository_, e.g., ``git push upstream v1.2.3``
#. Clean up the ``dist`` and ``build`` folders with ``tox -e clean``
   (or ``rm -rf dist build``)
   to avoid confusion with old builds and Sphinx docs.
#. Run ``tox -e build`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or git_ hash) according to the git_ tag.
   Also check the sizes of the distributions, if they are too big (e.g., >
   500KB), unwanted clutter may have been accidentally included.
#. Run ``tox -e publish -- --repository pypi`` and check that everything was
   uploaded to PyPI_ correctly.


.. |the repository service| replace:: GitLab
.. |contribute button| replace:: "Create merge request"
.. |project-name| replace:: ``symaware.base``

.. _repository: https://gitlab.mpi-sws.org/sadegh/eicsymaware
.. _issue tracker: https://gitlab.mpi-sws.org/sadegh/eicsymaware

.. |virtualenv| replace:: ``virtualenv``
.. |tox| replace:: ``tox``


.. _black: https://pypi.org/project/black/
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _flake8: https://flake8.pycqa.org/en/stable/
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _pip-venv: https://docs.python.org/3/library/venv.html
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _PyScaffold's contributor's guide: https://pyscaffold.org/en/stable/contributing.html
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _Numpy style: https://numpydoc.readthedocs.io/en/latest/format.html

.. _GitLab's IDE: https://docs.gitlab.com/ee/user/project/web_ide/
.. _GitLab's code editor: https://docs.gitlab.com/ee/user/project/repository/web_editor.html
