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

			const divWidget = node.widgets.find((w) => w.name === "divisor");
			divWidget.label = "multiple";

			const widthWidget = node.addWidget("number", "width", 0, () => {}, {
				precision: 0,
				serialize: false,
			});
			const heightWidget = node.addWidget("number", "height", 0, () => {}, {
				precision: 0,
				serialize: false,
			});
			widthWidget.disabled = true;
			heightWidget.disabled = true;
		} else if (node?.comfyClass === "LPIPSRun") {
			const lossWidget = node.addWidget("number", "image_loss", 0, () => {}, {
				precision: 6,
				serialize: false,
			});
			lossWidget.disabled = true;
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
			// The moment we add a 3rd widget, the text box gets compressed.
			// Preserve the original size.
			const sz = [...node.size];
			const loadWidget = node.addWidget(
				"button",
				"load",
				"load",
				() => {
					node.updatePreset();
				},
				{
					serialize: false,
				},
			);
			loadWidget.label = "📥load preset";
			node.setSize(sz);
			node.setDirtyCanvas(true, true);
		} else if (node?.comfyClass === "ImageDimensions") {
			const widthWidget = node.addWidget("number", "width", 0, () => {}, {
				precision: 0,
				serialize: false,
			});
			const heightWidget = node.addWidget("number", "height", 0, () => {}, {
				precision: 0,
				serialize: false,
			});
			widthWidget.disabled = true;
			heightWidget.disabled = true;
		} else if (node?.comfyClass === "ListCounter") {
			const countWidget = node.addWidget("number", "length", 0, () => {}, {
				precision: 0,
				serialize: false,
			});
			countWidget.disabled = true;
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
			} else if (node?.comfyClass === "ImageDimensions" || node?.comfyClass === "ResolutionChooser") {
				const widthWidget = node.widgets.find((w) => w.name === "width");
				const heightWidget = node.widgets.find((w) => w.name === "height");

				const values = event.detail.output.dims;
				const [width, height] = values[values.length - 1];
				widthWidget.value = width;
				heightWidget.value = height;
			} else if (node?.comfyClass === "IntegerLatch" || node?.comfyClass === "FloatLatch") {
				const valueWidget = node.widgets.find((w) => w.name === "value");
				const values = event.detail.output.value;
				valueWidget.value = values[values.length - 1];
			} else if (node?.comfyClass === "ListCounter") {
				const countWidget = node.widgets.find((w) => w.name === "length");
				const counts = event.detail.output.count;
				countWidget.value = counts[counts.length - 1];
			}
		});
	},
});
