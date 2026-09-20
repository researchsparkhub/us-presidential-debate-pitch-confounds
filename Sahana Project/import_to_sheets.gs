/**
 * Sahana Project — CSV → Google Sheets importer.
 *
 * Turns each per-debate CSV into its own formatted Google Sheet with:
 *   - frozen header row + frozen unit_id/speaker/text columns
 *   - a locked "Codebook" tab (DAMSL + Beads codes with definitions)
 *   - dropdown validation on the annotator tag columns
 *       (Phase A columns -> DAMSL codes, Phase B columns -> Beads codes)
 *   - warning-level protection on the source columns + codebook tab
 *
 * SETUP
 *   1. In Google Drive, create a folder named INPUT_FOLDER (below) and upload
 *      all the debate CSVs from Sahana Project/debates/ into it.
 *   2. Go to https://script.google.com  ->  New project  ->  paste this file.
 *   3. Run importAll().  Approve the permission prompt on first run.
 *   4. Formatted Sheets appear in the OUTPUT_FOLDER; share that folder with
 *      your annotators (Editor access).
 *
 * Re-running is safe: a debate whose Sheet already exists is skipped.
 */

// ===== CONFIG ===============================================================
var INPUT_FOLDER    = 'Sahana Debates - CSV';     // top Drive folder you created
var INPUT_SUBFOLDER = 'debates';                  // subfolder holding the CSVs; '' if CSVs
                                                  // sit directly inside INPUT_FOLDER
var OUTPUT_FOLDER   = 'Sahana Debates - Sheets';  // where formatted Sheets are created
var N_ANNOTATORS    = 3;

// ===== CODEBOOK (embedded — no upload needed) ===============================
var CODEBOOK = [
  // phase, code, name, definition
  ['Phase A (DAMSL)','Q-W','Wh-question','Question opening with what/where/when/why/how/who/which.'],
  ['Phase A (DAMSL)','YNQ','Yes-no question','Opens with an auxiliary verb; expects yes or no.'],
  ['Phase A (DAMSL)','Q-RHET','Rhetorical question','Not seeking an answer; often declarative + tag ("..., right?").'],
  ['Phase A (DAMSL)','OQ','Open-ended question','Invites an extended answer ("what do you think/propose?").'],
  ['Phase A (DAMSL)','SEEP','Seek explanation','Asks for reasoning ("why is...", "please explain/elaborate").'],
  ['Phase A (DAMSL)','S','Statement','Plain declarative assertion; default for a declarative sentence.'],
  ['Phase A (DAMSL)','S-Inform','Informative statement','Statement with new factual content — numbers, %, dollars, years.'],
  ['Phase A (DAMSL)','IAFA','Action directive / command','Imperative telling the listener to do something.'],
  ['Phase A (DAMSL)','IAFA-OF','Offer','Offers to do/provide something ("would you like", "I can offer").'],
  ['Phase A (DAMSL)','ACK','Acknowledgment','Backchannel ("I see", "right", "okay", "mm-hmm", "got it").'],
  ['Phase A (DAMSL)','AGR','Agreement','Explicit agreement ("I agree", "exactly", "you\'re right").'],
  ['Phase A (DAMSL)','DIS','Disagreement','Explicit disagreement ("I disagree", "that\'s not right").'],
  ['Phase A (DAMSL)','REJ','Rejection','Flat rejection ("no", "absolutely not", "that\'s false").'],
  ['Phase A (DAMSL)','ANS','Answer','Direct answer, typically opening with yes/no.'],
  ['Phase A (DAMSL)','GR','Greeting','Greeting/welcome ("good evening", "welcome", "thanks for joining").'],
  ['Phase A (DAMSL)','APO','Apology','Apology ("I\'m sorry", "my apologies", "I regret").'],
  ['Phase A (DAMSL)','THK','Thanks','Expression of thanks ("thank you", "thanks").'],
  ['Phase A (DAMSL)','EXPL','Explanation','Gives reasoning ("because", "the reason is", "in other words").'],
  ['Phase A (DAMSL)','CORR','Correction / self-clarification','Corrects own prior statement ("to be clear", "I misspoke").'],
  ['Phase A (DAMSL)','CH','Challenge / fact-check','Challenges accuracy ("that\'s incorrect/misleading/false").'],
  ['Phase A (DAMSL)','TT','Take/hold turn','Bids to hold the floor ("let me say", "if I may", "may I").'],
  ['Phase A (DAMSL)','TG','Turn-give','Hands the floor over ("what do you think?", "your turn").'],
  ['Phase A (DAMSL)','T-REQ','Turn request','Requests permission to speak ("can I respond/finish").'],
  ['Phase A (DAMSL)','R-REQ','Repeat request','Asks for repetition ("can you repeat", "say that again").'],
  ['Phase A (DAMSL)','MS','Maintain / steer topic','Steers back to a topic ("as I was saying", "I\'ll come back to").'],
  ['Phase A (DAMSL)','HS','Historical / self reference','References a year, historical figure, or event.'],
  ['Phase A (DAMSL)','INT','Interrupt','Cuts off another speaker. Needs timing — model leaves 0; human required.'],
  ['Phase A (DAMSL)','TA','Turn-accept','Accepts an offered turn. Contextual — model leaves 0; human required.'],
  ['Phase B (Beads)','AF','Appeal to fear','Fear/threat language (disaster, crisis, danger, invasion, collapse).'],
  ['Phase B (Beads)','AE','Appeal to emotion','Emotional appeal ("imagine if", "the children", "our families").'],
  ['Phase B (Beads)','AP','Appeal to pride','Group pride ("great country", "American values", "proud to be").'],
  ['Phase B (Beads)','APAT','Appeal to patriotism','Patriotic framing (freedom, liberty, constitution, flag).'],
  ['Phase B (Beads)','UF','Unity framing','Collective unity ("together we", "we as a nation", "common good").'],
  ['Phase B (Beads)','DF','Deflection','Redirects away ("the real question is", "let\'s not forget").'],
  ['Phase B (Beads)','PB','Political-outgroup bias','Negative outgroup framing ("the radical left", "the swamp").'],
  ['Phase B (Beads)','PER','Personal attack','Attacks character (opponent + liar/corrupt/incompetent).'],
  ['Phase B (Beads)','IT','Self-promotion / taking credit','Claims credit ("I created", "I built", "I delivered").'],
  ['Phase B (Beads)','IP','Injecting personal stance','Frames as opinion ("I believe", "in my view", "I contend").'],
  ['Phase B (Beads)','BQ','Biased / loaded question','Question containing loaded terms (liar, corrupt, fraud).'],
  ['Phase B (Beads)','ATTR','Blame attribution','Blames a named opponent ("destroyed", "failed us").'],
  ['Phase B (Beads)','AEX','Adversarial exchange','Personal attack + question or emphatic jab.'],
  ['Phase B (Beads)','REB','Rebuttal','Challenge/disagreement that references the opponent.'],
  ['Phase B (Beads)','RB','Race-related content','Mentions race/ethnicity (racial, ethnic, community terms).'],
  ['Phase B (Beads)','GB','Gender bias','Bias based on gender. Needs judgement — model leaves 0; human required.'],
  ['Phase B (Beads)','GD','Gender dismissal','Dismissing on gendered grounds. Contextual — model leaves 0; human.'],
  ['Phase B (Beads)','CB','Cognitive bias','Reasoning bias (false dilemma etc.). Model leaves 0; human required.'],
  ['Phase B (Beads)','IA','Issue avoidance','Avoids the substantive issue. Model leaves 0; human required.'],
  ['Phase B (Beads)','SE','Selective evidence','Cherry-picks facts. Needs world knowledge — model leaves 0; human.'],
  ['Phase B (Beads)','CBias','Cultural bias','Bias tied to culture/group. Contextual — model leaves 0; human.']
];

function damslCodes_() {
  return CODEBOOK.filter(function(r){ return r[0].indexOf('DAMSL') > -1; }).map(function(r){ return r[1]; });
}
function beadsCodes_() {
  return CODEBOOK.filter(function(r){ return r[0].indexOf('Beads') > -1; }).map(function(r){ return r[1]; });
}

// ===== MAIN =================================================================
function importAll() {
  var inFolder = resolveInputFolder_();
  var outFolder = getOrCreateFolder_(OUTPUT_FOLDER);

  var existing = {};
  var it = outFolder.getFiles();
  while (it.hasNext()) { existing[it.next().getName()] = true; }

  var files = inFolder.getFilesByType(MimeType.CSV);
  var made = 0, skipped = 0;
  while (files.hasNext()) {
    var f = files.next();
    var name = f.getName().replace(/\.csv$/i, '');
    if (name.toLowerCase() === 'codebook') continue;
    if (existing[name]) { skipped++; Logger.log('skip (exists): ' + name); continue; }
    buildSheet_(f, name, outFolder);
    made++;
    Logger.log('built: ' + name);
  }
  Logger.log('Done. Built ' + made + ', skipped ' + skipped + '. Output folder: ' + OUTPUT_FOLDER);
}

function buildSheet_(csvFile, name, outFolder) {
  var data = Utilities.parseCsv(csvFile.getBlob().getDataAsString('UTF-8'));
  if (data.length === 0) return;

  var ss = SpreadsheetApp.create(name);
  DriveApp.getFileById(ss.getId()).moveTo(outFolder);

  var sh = ss.getSheets()[0];
  sh.setName('Annotation');
  sh.getRange(1, 1, data.length, data[0].length).setValues(data);

  // Header styling + freezes
  sh.setFrozenRows(1);
  sh.setFrozenColumns(3);
  var header = sh.getRange(1, 1, 1, data[0].length);
  header.setFontWeight('bold').setBackground('#1a237e').setFontColor('#ffffff').setWrap(true);
  sh.getRange(1, 1, data.length, 1).setHorizontalAlignment('center'); // unit_id
  sh.setColumnWidth(2, 130);   // speaker
  sh.setColumnWidth(3, 520);   // text
  sh.getRange(2, 3, data.length - 1, 1).setWrap(true);

  // Column tinting + dropdowns for each annotator block
  var damsl = damslCodes_(), beads = beadsCodes_();
  var vaA = SpreadsheetApp.newDataValidation().requireValueInList(damsl, true).setAllowInvalid(true)
             .setHelpText('DAMSL code(s). For multiple, separate with "; ".').build();
  var vaB = SpreadsheetApp.newDataValidation().requireValueInList(beads, true).setAllowInvalid(true)
             .setHelpText('Beads code(s). For multiple, separate with "; ".').build();
  var tints = ['#e8f0fe', '#e6f4ea', '#fce8e6']; // per-annotator background
  var nRows = data.length - 1;
  for (var a = 0; a < N_ANNOTATORS; a++) {
    var base = 4 + a * 4;                 // 1-indexed: Phase A col of annotator a+1
    var tint = tints[a % tints.length];
    sh.getRange(1, base, data.length, 4).setBackground(tint);
    sh.getRange(1, base, 1, 4).setBackground('#1a237e').setFontColor('#ffffff').setFontWeight('bold').setWrap(true);
    if (nRows > 0) {
      sh.getRange(2, base,     nRows, 1).setDataValidation(vaA); // Phase A (DAMSL)
      sh.getRange(2, base + 2, nRows, 1).setDataValidation(vaB); // Phase B (Beads)
    }
    sh.setColumnWidth(base, 130); sh.setColumnWidth(base + 1, 200);
    sh.setColumnWidth(base + 2, 130); sh.setColumnWidth(base + 3, 200);
  }

  // Protect source columns (warning-only, so no editor management needed)
  var pSrc = sh.getRange(1, 1, data.length, 3).protect()
              .setDescription('Source columns — do not edit');
  pSrc.setWarningOnly(true);

  addCodebookSheet_(ss);
  addReadmeSheet_(ss, name);
}

function addCodebookSheet_(ss) {
  var cb = ss.insertSheet('Codebook');
  cb.getRange(1, 1, 1, 4).setValues([['Phase', 'Code', 'Name', 'Definition']])
    .setFontWeight('bold').setBackground('#1a237e').setFontColor('#ffffff');
  cb.getRange(2, 1, CODEBOOK.length, 4).setValues(CODEBOOK);
  cb.setFrozenRows(1);
  cb.setColumnWidth(1, 130); cb.setColumnWidth(2, 90);
  cb.setColumnWidth(3, 220); cb.setColumnWidth(4, 520);
  cb.getRange(2, 4, CODEBOOK.length, 1).setWrap(true);
  var p = cb.protect().setDescription('Codebook — reference only'); p.setWarningOnly(true);
}

function addReadmeSheet_(ss, name) {
  var r = ss.insertSheet('READ ME', 0); // first tab
  var lines = [
    ['Debate: ' + name],
    [''],
    ['HOW TO VERIFY'],
    ['1. Go to the Annotation tab.'],
    ['2. Each row is one sentence. Columns unit_id / speaker / text are locked context.'],
    ['3. Your Phase A (DAMSL) and Phase B (Beads) columns are PRE-FILLED with the model\'s tags.'],
    ['4. Read the sentence and VERIFY the pre-filled tag:'],
    ['     - Correct  -> leave it.'],
    ['     - Wrong    -> replace it with the right code (dropdown) and say why in the notes column.'],
    ['     - Missing a tag -> add it (separate multiple codes with "; ", e.g.  S; ATTR ).'],
    ['     - Propose a NEW tag not in the codebook -> type it and explain in the notes column.'],
    ['5. Codes and their meanings are on the Codebook tab.'],
    [''],
    ['Only edit YOUR annotator columns. Phase A = DAMSL (discourse act). Phase B = Beads (bias).'],
    ['Some codes (INT, TA, GB, GD, CB, IA, SE, CBias) are never auto-filled and rely on your judgement.']
  ];
  r.getRange(1, 1, lines.length, 1).setValues(lines);
  r.getRange(1, 1).setFontWeight('bold').setFontSize(14);
  r.getRange(3, 1).setFontWeight('bold');
  r.setColumnWidth(1, 720);
}

function getOrCreateFolder_(folderName) {
  var it = DriveApp.getFoldersByName(folderName);
  return it.hasNext() ? it.next() : DriveApp.createFolder(folderName);
}

/**
 * Resolve the folder that holds the CSVs by navigating INPUT_FOLDER -> INPUT_SUBFOLDER.
 * DriveApp searches by NAME (not path), so we look up the top folder, then find the
 * named subfolder *inside it* — this disambiguates a common name like "debates".
 */
function resolveInputFolder_() {
  var tops = DriveApp.getFoldersByName(INPUT_FOLDER);
  if (!tops.hasNext()) {
    throw new Error('Folder "' + INPUT_FOLDER + '" not found in Drive. Create it and upload the CSVs.');
  }
  var top = tops.next();
  if (tops.hasNext()) {
    throw new Error('More than one folder named "' + INPUT_FOLDER + '" exists — rename or remove the extras so it is unique.');
  }
  if (!INPUT_SUBFOLDER) return top;

  var subs = top.getFoldersByName(INPUT_SUBFOLDER);
  if (!subs.hasNext()) {
    throw new Error('Subfolder "' + INPUT_SUBFOLDER + '" not found inside "' + INPUT_FOLDER +
                    '". If the CSVs sit directly in "' + INPUT_FOLDER + '", set INPUT_SUBFOLDER = \'\'.');
  }
  return subs.next();
}
