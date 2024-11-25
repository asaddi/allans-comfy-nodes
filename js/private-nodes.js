import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
app.registerExtension({
	name: "private.nodes",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeType.comfyClass === "PresetText") {
			nodeType.prototype.updatePreset = function (name) {
				const presetWidget = this.widgets.find((w) => w.name === "preset");
				const textWidget = this.widgets.find((w) => w.name === "text");

				const presetName = name ?? presetWidget.value;

				if (presetName === "None") {
					textWidget.value = "";
				} else {
					api.fetchApi(`/preset_text?name=${presetName}`).then((resp) => {
						resp.json().then((data) => {
							textWidget.value = data;
						});
					});
				}
			};
		}
	},

	async nodeCreated(node) {
		if (node?.comfyClass === "ResolutionChooser") {
			const pixelsWidget = node.widgets.find((w) => w.name === "megapixels");
			pixelsWidget.label = "mebipixels";
		} else if (node?.comfyClass === "LPIPSRun") {
			node.addWidget("number", "image_loss", 0, () => {}, {
				// TODO how to make read only??
				serialize: false,
			});
		} else if (node?.comfyClass === "PrivateSeed") {
			node.properties.randomizeSeed = true;

			// rename seed widget
			const seedWidget = node.widgets.find((w) => w.name === "seed_value");
			seedWidget.label = "seed";

			const newSeed = () => {
				seedWidget.value = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
			};
			newSeed();

			// add extra buttons
			const randomWidget = node.addWidget(
				"button",
				"randomize",
				"randomize",
				() => {
					node.properties.randomizeSeed = true;
					newSeed();
				},
				{
					serialize: false,
				},
			);
			randomWidget.label = "🎲always randomize";

			const fixedWidget = node.addWidget(
				"button",
				"fixed",
				"fixed",
				() => {
					node.properties.randomizeSeed = false;
					newSeed();
				},
				{
					serialize: false,
				},
			);
			fixedWidget.label = "🎲new fixed seed";

			const histWidget = node.addWidget(
				"button",
				"previous",
				"previous",
				function () {
					node.properties.randomizeSeed = false;
					seedWidget.value = this.value;
				},
				{
					serialize: false,
				},
			);
			histWidget.label = "♻️previous seed";
			histWidget.disabled = true;
		} else if (node?.comfyClass === "PresetText") {
			const presetWidget = node.widgets.find((w) => w.name === "preset");

			const original_callback = presetWidget.callback;
			presetWidget.callback = function (...args) {
				const name = args?.[0];
				node.updatePreset(name);
				return original_callback?.apply(this, args);
			};

			const textWidget = node.widgets.find((w) => w.name === "text");
			textWidget.inputEl.readOnly = true;
		}
	},

	async setup() {
		api.addEventListener("executed", (event) => {
			const node = app.graph.getNodeById(event.detail.node);
			if (node?.comfyClass === "LPIPSRun") {
				const lossWidget = node.widgets.find(
					(widget) => widget.name === "image_loss",
				);
				const image_losses = event.detail.output.image_loss;
				lossWidget.value = image_losses[image_losses.length - 1];
				app.graph.setDirtyCanvas(true, false);
			} else if (node?.comfyClass === "PrivateSeed") {
				const histWidget = node.widgets.find((w) => w.name === "previous");

				const values = event.detail.output.seed_value;
				histWidget.value = values[values.length - 1];
				histWidget.label = `♻️${histWidget.value}`;
				histWidget.disabled = false;

				const seedWidget = node.widgets.find((w) => w.name === "seed_value");
				if (node.properties.randomizeSeed) {
					seedWidget.value = Math.floor(
						Math.random() * Number.MAX_SAFE_INTEGER,
					);
				}
			}
		});
	},
});
