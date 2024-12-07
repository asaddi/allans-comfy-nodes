// Copyright (c) 2024 Allan Saddi <allan@saddi.com>
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
		} else if (nodeType.comfyClass === "FloatListStepSize") {
			nodeType.prototype.updateStepCount = function () {
				const startWidget = this.widgets.find((w) => w.name === "start");
				const endWidget = this.widgets.find((w) => w.name === "end");
				const sizeWidget = this.widgets.find((w) => w.name === "step_size");

				if (sizeWidget.value != 0) {
					const steps = Math.abs(endWidget.value - startWidget.value) / Math.abs(sizeWidget.value);
					this.calculated_steps = Math.ceil(steps) + 1;  // start & end guaranteed
				}
			}
		}
	},

	async nodeCreated(node) {
		if (node?.comfyClass === "ResolutionChooser") {
			const pixelsWidget = node.widgets.find((w) => w.name === "megapixels");
			pixelsWidget.label = "mebipixels";

			const divWidget = node.widgets.find((w) => w.name === "divisor");
			divWidget.label = "multiple";

			const makeBadge = () => {
				return new LGraphBadge({
					text: `${node.calculated_width ?? "?"}\u00d7${node.calculated_height ?? "?"}`
				});
			}
			node.badges.push(makeBadge);
		} else if (node?.comfyClass === "LPIPSRun") {
			const makeBadge = () => {
				return new LGraphBadge({
					text: `loss: ${node.properties?.loss?.toFixed(6) ?? "?"}`
				});
			}
			node.badges.push(makeBadge);
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

			randomWidget.afterQueued = () => {
				if (node.properties.randomizeSeed) {
					newSeed();
					randomWidget.callback(randomWidget.value);
				}
			};

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
			fixedWidget.label = "🔒new fixed seed";

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

			const makeBadge = () => {
				return new LGraphBadge({
					text: node.properties.randomizeSeed ? "🎲" : "🔒",
				});
			};
			node.badges.push(makeBadge);
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
			const makeBadge = () => {
				return new LGraphBadge({
					text: `${node.calculated_width ?? "?"}\u00d7${node.calculated_height ?? "?"}`
				});
			}
			node.badges.push(makeBadge);
		} else if (node?.comfyClass === "ListCounter") {
			const makeBadge = () => {
				return new LGraphBadge({
					text: `count: ${node.element_count ?? "?"}`
				});
			}
			node.badges.push(makeBadge);
		} else if (node?.comfyClass === "FloatListStepSize") {
			const startWidget = node.widgets.find((w) => w.name === "start");
			const endWidget = node.widgets.find((w) => w.name === "end");
			const sizeWidget = node.widgets.find((w) => w.name === "step_size");

			const startWidget_callback = startWidget.callback;
			startWidget.callback = function (...args) {
				node.updateStepCount();
				return startWidget_callback?.apply(this, args);
			}

			const endWidget_callback = endWidget.callback;
			endWidget.callback = function (...args) {
				node.updateStepCount();
				return endWidget_callback?.apply(this, args);
			}

			const sizeWidget_callback = sizeWidget.callback;
			sizeWidget.callback = function (...args) {
				node.updateStepCount();
				return sizeWidget_callback?.apply(this, args);
			}

			const makeBadge = () => {
				return new LGraphBadge({
					text: `steps: ${node.calculated_steps ?? "?"}`
				});
			}
			node.badges.push(makeBadge);
		}
	},

	async setup() {
		api.addEventListener("executed", (event) => {
			const node = app.graph.getNodeById(event.detail.node);
			if (node?.comfyClass === "LPIPSRun") {
				const image_losses = event.detail.output.image_loss;
				node.properties.loss = image_losses[image_losses.length - 1];
			} else if (node?.comfyClass === "PrivateSeed") {
				const histWidget = node.widgets.find((w) => w.name === "previous");

				const values = event.detail.output.seed_value;
				histWidget.value = values[values.length - 1];
				histWidget.label = `♻️${histWidget.value}`;
				histWidget.disabled = false;
			} else if (
				node?.comfyClass === "ImageDimensions" ||
				node?.comfyClass === "ResolutionChooser"
			) {
				const values = event.detail.output.dims;
				const [width, height] = values[values.length - 1];
				node.calculated_width = width;
				node.calculated_height = height;
			} else if (
				node?.comfyClass === "IntegerLatch" ||
				node?.comfyClass === "FloatLatch"
			) {
				const valueWidget = node.widgets.find((w) => w.name === "value");
				const values = event.detail.output.value;
				valueWidget.value = values[values.length - 1];
			} else if (node?.comfyClass === "ListCounter") {
				const counts = event.detail.output.count;
				node.element_count = counts[counts.length - 1];
			}
		});
	},
});
