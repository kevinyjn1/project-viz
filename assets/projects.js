// The list of projects. The top bar and the home page are built from this list.
// To add a project: add an entry here and make a folder with the same `slug`.
//   slug: folder name   title: card title   short: top-bar tab label   blurb: one line
//   model: the kind of model   repo: GitHub link   team: optional credit line on the card
export const PROJECTS = [
  {
    slug: 'hydrosense',
    title: 'HydroSense',
    short: 'HydroSense',
    blurb: 'Tap a water bottle, hear how full it is: microphone audio, MFCC features and a linear SVM.',
    model: 'Linear SVM',
    repo: 'https://github.com/saideepk444/HydroSense',
    team: 'Team project with Saideep Kasipathy and Min Park',
  },
  {
    slug: 'lol-predictor',
    title: 'Checkpoint Champion',
    short: 'LoL predictor',
    blurb: 'Predicting who wins a League of Legends game from the state of the match at minute 25.',
    model: 'Logistic regression',
    repo: 'https://github.com/MrTuge123/League_of_Legends_Analysis',
    team: 'With Xingzhi Cui',
  },
  {
    slug: 'ambient-mnist',
    title: 'AmbientMNIST',
    short: 'AmbientMNIST',
    blurb: 'A small CNN reading handwritten digits on a solar-powered microcontroller that keeps losing power.',
    model: 'CNN',
    repo: 'https://github.com/kevinyjn1/AmbientMNIST',
  },
];
