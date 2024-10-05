const vscode = require('vscode');
const path = require('path');
const fs = require('fs');

let pathMD = null;
let pathComfy = null;
let pathPython = null;
let pathOutput = null;
let pathWorkflows = null;
let pathStyles = null;
let pathSplash = null;
let panel = undefined;
let term = undefined;

const WORKFLOWS = [
	"default_txt2img_api.json",
	"default_img2img_api.json",
	"default_inpaint_api.json",
	"default_inpaint_pro_api.json"
]

function setPaths(extPath) {
	pathComfy = vscode.workspace.getConfiguration('mental-diffusion').comfyPath;
	pathPython = vscode.workspace.getConfiguration('mental-diffusion').comfyPython;
	pathOutput = vscode.workspace.getConfiguration('mental-diffusion').output;
	pathWorkflows = vscode.workspace.getConfiguration('mental-diffusion').workflows;
	pathStyles = vscode.workspace.getConfiguration('mental-diffusion').styles;
	pathSplash = vscode.workspace.getConfiguration('mental-diffusion').splash;

	if (pathOutput == '.output')
		pathOutput = extPath + '/.output';
	if (pathWorkflows == '.workflows')
		pathWorkflows = extPath + '/.workflows';
	if (pathStyles == 'configs/styles.json')
		pathStyles = extPath + '/configs/styles.json';
	if (pathSplash == 'media/splash.png')
		pathSplash = extPath + '/media/splash.png';

	if (!fs.existsSync(pathComfy + "/main.py"))
		vscode.window.showErrorMessage("Mental Diffusion: Set ComfyUI Source path in settings.");
	if (!fs.existsSync(pathPython))
		vscode.window.showErrorMessage("Mental Diffusion: Set ComfyUI Python path in settings.");
	if (!fs.existsSync(pathOutput))
		vscode.window.showErrorMessage("Mental Diffusion: Set Output path in settings.");
	if (!fs.existsSync(pathWorkflows))
		vscode.window.showErrorMessage("Mental Diffusion: Set Workflows path in settings.");
	if (!fs.existsSync(pathStyles))
		vscode.window.showErrorMessage("Mental Diffusion: Set Styles path in settings.");
}

function activate(ctx) {
	pathMD = ctx.extensionPath;
	
	if (!fs.existsSync(pathMD + '/.input'))
		fs.mkdirSync(pathMD + '/.input');
	if (!fs.existsSync(pathMD + '/.output'))
		fs.mkdirSync(pathMD + '/.output');
	if (!fs.existsSync(pathMD + '/.workflows'))
		fs.mkdirSync(pathMD + '/.workflows');

	setPaths(pathMD);

	ctx.subscriptions.push(vscode.commands.registerCommand('md.start', () => {
		setPaths(pathMD);
		openTerminal();
		openWebInterface();
	}));

	ctx.subscriptions.push(vscode.commands.registerCommand('md.comfy', () => {
		setPaths(pathMD);
		openTerminal();
	}));

	ctx.subscriptions.push(vscode.commands.registerCommand('md.webui', () => {
		setPaths(pathMD);
		openWebInterface();
	}));
}

function openTerminal() {
	if (term) term.dispose();
	const args = vscode.workspace.getConfiguration('mental-diffusion').comfyArguments;
	term = vscode.window.createTerminal('ComfyUI');
	term.sendText('cls');
	term.sendText(`${pathPython} -s ${pathComfy}/main.py --enable-cors-header --preview-method auto ${args}`);
	term.show();
}

function openWebInterface() {
	if (panel) panel.dispose();
	panel = vscode.window.createWebviewPanel(
		'MD', 'Mental Diffusion', vscode.ViewColumn.One, {
			enableScripts: true,
			retainContextWhenHidden: true
		}
	);
	panel.webview.html = fs.readFileSync(pathMD + '/src/index.html').toString();
	panel.webview.html = panel.webview.html.replace('{SPLASH}', fs.readFileSync(pathSplash, { encoding: 'base64' }).toString());
	//panel.webview.html = panel.webview.html.replace('{SCRIPT}', panel.webview.asWebviewUri(vscode.Uri.file(pathMD + '/src/main.js')));
	panel.webview.onDidReceiveMessage(msg => {
		switch (msg.type) {
			case "initialize":
				panel.webview.postMessage({
					path_md: pathMD,
					path_comfy: pathComfy,
					workflows: getWorkflows().concat(getCustomWorkflows(pathWorkflows)),
					workflow_upscale: fs.readFileSync(`${ pathMD }/configs/upscale_api.json`).toString(),
					workflow_merge: fs.readFileSync(`${ pathMD }/configs/merge_api.json`).toString(),
					styles: fs.readFileSync(pathStyles).toString()
				});
				break;

			case "input": // save image to .input
				const input = path.join(pathMD + '/.input', msg.filename);
				fs.writeFileSync(input, decodeBase64Image(msg.base64));
				break;

			case "output": // save image to .output or custom path
				saveImage(path.join(pathOutput, msg.filename), msg.base64);
				break;

			case "saveas": // save image as...
				saveImageAs(path.join(pathOutput, msg.filename), msg.base64);
				break;

			case "newfile": // open content in a new file
				openUntitled(msg.content, msg.language);
				break;

			case "open_output":
				vscode.env.openExternal(pathOutput);
				break;

			case "open_workflows":
				vscode.env.openExternal(pathWorkflows);
				break;

			case "open_styles":
				vscode.commands.executeCommand('vscode.open', vscode.Uri.file(pathStyles));
				break;

			case "url":
				vscode.env.openExternal(vscode.Uri.parse(msg.url));
				break;
			
			case "alert":
				vscode.window.showInformationMessage(msg.message);
				break;
		}
	});
}

function getWorkflows() {
	const arr = [];
	for (let i = 0; i < WORKFLOWS.length; i++) {
		const filepath = `${ pathMD }/configs/${ WORKFLOWS[i] }`;
		arr.push([ WORKFLOWS[i], fs.readFileSync(filepath).toString() ]);
	}
	return arr;
}

function getCustomWorkflows(dir) {
	const arr = [];
	fs.readdirSync(dir).forEach((file) => {
		const filepath = dir + '/' + vscode.Uri.file(file).path;
		if (filepath.endsWith(".json"))
			arr.push([ vscode.Uri.file(file).path.substring(1), fs.readFileSync(filepath).toString() ]);
	});
	return arr;
}

function saveImage(fpath, base64) {
	if (fs.existsSync(path.dirname(fpath))) {
		fpath = incrementFilename(fpath);
		fs.writeFileSync(fpath, decodeBase64Image(base64));
	} else {
		vscode.window.showErrorMessage("The output directory does not exists");
	}
}

function saveImageAs(fpath, base64) {
	fpath = incrementFilename(fpath);
	vscode.window.showSaveDialog({
		defaultUri: vscode.Uri.file(fpath),
		filters: { 'Images': ['png'] }
	}).then(uri => {
		if (uri) fs.writeFileSync(uri.fsPath, decodeBase64Image(base64));
	});
}

async function openUntitled(content, language) {
    const document = await vscode.workspace.openTextDocument({ language, content });
    vscode.window.showTextDocument(document);
}

function incrementFilename(filepath) { // TODO: rare cases like *._#.ext and *._##.ext
	const ext = path.extname(filepath);
	const name = path.basename(filepath, ext);
	const ending = name.split('_').pop();
	let idx = 1;
	while (fs.existsSync(filepath)) {
		if (!isNaN(ending) && ending.length < 3) {
			idx += parseInt(ending);
			filepath = `${ path.dirname(filepath) }/${ name }_${ idx }${ ext }`;
		} else {
			filepath = `${ path.dirname(filepath) }/${ name }_${ idx }${ ext }`;
			idx++;
		}
	}
	return filepath;
}

function decodeBase64Image(b64) {
	return Buffer.from(b64.split(',')[1], 'base64');
}

function deactivate() {
	if (term) term.dispose();
	if (panel) panel.dispose();
}

module.exports = {
	activate,
	deactivate
}
