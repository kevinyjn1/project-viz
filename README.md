# Under the Hood

Interactive, step-by-step walkthroughs of what my machine-learning projects actually compute, with the real numbers at every step.

**Live site:** https://kevinyjn1.github.io/project-viz/

Each project gets its own page. On the left is a panel of text you move through with **Next** and **Back** (or the arrow keys, or a sideways swipe on a phone). On the right is a picture that changes with each step: the audio, the features, the weights, the prediction. The address bar remembers which step you are on (for example `.../hydrosense/#4`), so you can send someone a link to one exact step.

| Project | What it does | Model | Code |
|---|---|---|---|
| HydroSense | Tap a water bottle and hear how full it is | Linear SVM on MFCC audio features | [saideepk444/HydroSense](https://github.com/saideepk444/HydroSense) |
| Checkpoint Champion | Predict who wins a League of Legends game at minute 25 | Logistic regression | [MrTuge123/League_of_Legends_Analysis](https://github.com/MrTuge123/League_of_Legends_Analysis) |
| AmbientMNIST | Read handwritten digits on a solar-powered microcontroller that keeps losing power | Small CNN | [kevinyjn1/AmbientMNIST](https://github.com/kevinyjn1/AmbientMNIST) |

The site is plain HTML, CSS and JavaScript. There is no build step: the files in this folder are exactly what the browser gets.

## Folder layout

```
project-viz/
├── index.html          the home page (one card per project)
├── 404.html            shown for any address that doesn't exist
├── .nojekyll           tells GitHub Pages to serve the files as they are
├── assets/             shared by every page
│   ├── site.css        colors, fonts, the top bar and the panel + stage layout
│   ├── home.css        extra styles for the home page and the 404 page
│   ├── stepper.js      the Next/Back walkthrough engine every project page uses
│   ├── projects.js     THE LIST OF PROJECTS (the home page and the top tabs are built from it)
│   └── previews.js     the small moving pictures on the home page cards
├── hydrosense/         one folder per project
│   ├── index.html      the page itself (short; it loads viz.js)
│   ├── viz.js          the steps: their text and what to draw for each one
│   └── data/           numbers worked out ahead of time (JSON files)
├── lol-predictor/      same shape
├── ambient-mnist/      same shape
└── tools/              Python scripts that produce each project's data/ files
    ├── hydrosense/
    ├── lol-predictor/
    └── ambient-mnist/
```

## Preview the site on your own computer

You need a tiny local web server, because browsers refuse to load JavaScript modules straight from a file on disk. Python has one built in.

1. Open a terminal.
2. Go into the site folder: `cd path/to/project-viz`
3. Start the server: `python3 -m http.server`
4. Open http://localhost:8000 in your browser.
5. After you change a file, save it and reload the page in the browser. Stop the server with `Ctrl+C`.

## Change the text of a step

Every step's words live in that project's `viz.js`, in a list called `chapters`. It looks like this:

```js
const chapters = [
  { title: 'Audio', steps: [
    { title: 'One tap',
      body: '<p>The words shown in the panel. This is HTML, so use <p> for paragraphs.</p>',
      enter(ctx) { /* draws the picture for this step */ } },
    // ...more steps
  ]},
  // ...more chapters
];
```

To fix wording, change the `title` or `body` text and save. Leave `enter(...)` alone unless you want to change the picture. Adding a new step means adding another `{ title, body, enter }` block in the right place in the list.

## Add a new project

1. Open `assets/projects.js` and add an entry to the list:
   ```js
   {
     slug: 'my-project',          // folder name, letters, numbers and dashes only
     title: 'My Project',         // big title on the home page card
     short: 'My Project',         // short label for the tab in the top bar
     blurb: 'One sentence about what it does.',
     model: 'Random forest',      // the small tag on the card
     repo: 'https://github.com/kevinyjn1/my-project',
     team: 'With ...',            // optional credit line, leave it out if it's just you
   },
   ```
2. Copy an existing project folder (for example `lol-predictor/`) and rename the copy to the same `slug`.
3. In the copy's `index.html` and `viz.js`, change the title, credit, GitHub link and the steps.
4. Preview it locally (see above). The new card and tab appear by themselves.
5. Optional: give the card its own moving preview by adding a drawing function in `assets/previews.js` (the comments at the top say how). Without one, the card gets a simple generic animation.

## Recompute the numbers

The pages don't run the models in the browser. Instead, a Python script in `tools/<slug>/` loads the project's real data and model, works out the numbers each step shows, and saves them as JSON in `<slug>/data/`. You only need to rerun a script if you change the data, the model, or what a step should show.

```bash
cd path/to/project-viz
python3 tools/lol-predictor/precompute.py      # writes lol-predictor/data/*.json
```

Each script says at the top what it needs (for example a copy of the project's own repo, and Python packages such as numpy or scikit-learn) and where it writes its files. Reload the page afterwards to see the new numbers.

## Publish with GitHub Pages

1. Create a repository on GitHub (for example `project-viz`) and push this folder to its `main` branch.
2. On GitHub, open the repository's **Settings > Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**, then pick branch **main** and folder **/ (root)**, and press **Save**.
4. After a minute or two the site is live at `https://kevinyjn1.github.io/<repository-name>/`. Every later push to `main` updates it automatically.

All links inside the site are relative, so the repository can have any name.

## Credits

- **HydroSense**: a team project by Yunjong (Kevin) Na, Saideep Kasipathy and Min Park. Code: [saideepk444/HydroSense](https://github.com/saideepk444/HydroSense)
- **Checkpoint Champion**: by Xingzhi Cui and Yun Jong (Kevin) Na. Code: [MrTuge123/League_of_Legends_Analysis](https://github.com/MrTuge123/League_of_Legends_Analysis)
- **AmbientMNIST**: [kevinyjn1/AmbientMNIST](https://github.com/kevinyjn1/AmbientMNIST)

Made by Kevin (Yun Jong) Na · [github.com/kevinyjn1](https://github.com/kevinyjn1) · also see [Inside SmolVLA](https://kevinyjn1.github.io/smolvla-viz-site/), a 3D walk through a robot policy.
