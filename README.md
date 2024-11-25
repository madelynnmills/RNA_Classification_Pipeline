# RNA_Classification_Pipeline
Machine learning pipeline for RNA classification and motif analysis


# Setting Up Your Python Package
Make sure to use A Python Version > 3.10.6

The following will help you set up a Python package step by step.

## What is a Virtual Environment?

A virtual environment is like a special folder for your Python project. It keeps all the tools and libraries your project needs separate from everything else on your computer. This avoids problems like one project messing up another.

For more details, see: [How to Setup Virtual Environments in Python](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).

---

## Installing Homebrew

[Homebrew](https://brew.sh/) is a tool that helps you install other tools on your computer easily.

1. Open your **Terminal**. You can find this by searching for "Terminal" on your Mac or Linux computer.
2. Copy and paste this command into the Terminal, then press **Enter**:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. When it’s finished, type this to make sure it’s installed:

   ```bash
   brew --version
   ```

---

## Installing pyenv

`pyenv` helps you install and manage different versions of Python.

1. Install `pyenv` by running these commands:

   ```bash
   brew update
   brew install pyenv
   ```

2. Now, you need to tell your computer how to use `pyenv`. This involves editing a file that your terminal reads when it starts.

   ### What Are `bash` and `zsh`?

   - **`bash`** and **`zsh`** are programs that control how your terminal works. Think of them as different styles of the same tool.
   - To find out which one you’re using, type this in the Terminal and press **Enter**:

     ```bash
     echo $SHELL
     ```

     If it says something like `/bin/bash`, you’re using bash. If it says `/bin/zsh`, you’re using zsh.

3. Edit the shell configuration file based on which one you’re using:

   - If you’re using **bash**, the file is named `~/.bashrc`.
   - If you’re using **zsh**, the file is named `~/.zshrc`.

   ### How to Edit Your Shell File
   You can use a text editor to open and edit this file. Here’s how:

   - **If you’re using Visual Studio Code:**
     Open the file with this command:
     ```bash
     code ~/.bashrc  # For bash
     code ~/.zshrc   # For zsh
     ```

   - **If you’re using PyCharm:**
     Open the file with this command:
     ```bash
     pycharm ~/.bashrc  # For bash
     pycharm ~/.zshrc   # For zsh
     ```

4. Add the following lines to the file:

   ```bash
   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init --path)"
   ```

5. Save the file and close the editor.

6. Apply the changes to your terminal by running one of these commands:

   ```bash
   source ~/.bashrc  # For bash
   source ~/.zshrc   # For zsh
   ```

7. Verify that `pyenv` is installed and working:

   ```bash
   pyenv --version
   ```

---

## Installing and Managing Python Versions

1. List available Python versions:

   ```bash
   pyenv install --list
   ```

2. Install a specific Python version:

   ```bash
   pyenv install <version>
   ```

   Example:

   ```bash
   pyenv install 3.11.6
   ```

3. Make this the default Python version:

   ```bash
   pyenv global 3.11.6
   ```

4. Use this version only in a specific folder:

   ```bash
   pyenv local 3.11.6
   ```

5. Check your current Python version:

   ```bash
   python --version
   ```

---

## Setting Up Your Python Package

### 1. Create a Virtual Environment
Run this command to create a virtual environment for your project:

```bash
python3 -m venv venv
```

This creates a folder called `venv`.

### 2. Activate the Virtual Environment
Before you work on your project, activate the virtual environment:

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

Once activated, your terminal will show `(venv)` before the prompt.

### 3. Install Project Dependencies
If your project has a `requirements.txt` file, install its dependencies:

```bash
pip install -r requirements.txt
```

### 4. Adding New Packages
To add a new package, use:

```bash
pip install <package-name>
```

> **Important:** After installing a new package, update your `requirements.txt` file with:

```bash
pip freeze > requirements.txt
```

### 5. Deactivate the Virtual Environment
When you’re done working, deactivate the virtual environment:

```bash
deactivate
```

---

## How to Run the App

Once you have set up your environment and installed all the required packages, you can run the app with the following command:

```bash
python main.py
```

Make sure your virtual environment is activated before running this command (you’ll see `(venv)` in your terminal prompt if it’s active). If you’re not sure, activate it again:

```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

This will start the app using the `main.py` file. If there are any issues, double-check that all dependencies are installed with:

```bash
pip install -r requirements.txt
```
