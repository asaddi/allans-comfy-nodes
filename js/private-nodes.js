import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({
	name: "private.nodes",
	async nodeCreated(node) {
		if (node?.comfyClass === "LPIPSRun") {
			node.addWidget(
				"number",
				"image_loss",
				0,
				function() {},
				{
					// TODO how to make read only??
					serialize: false,
				}
			);
		}

		if (node?.comfyClass === "PrivateSeed") {
			// rename seed widget
			const seedWidget = node.widgets.find((w) => w.name === "seed_value");
			seedWidget.label = "seed";

			// add extra buttons
			node.addWidget(
				"button",
				"🎲always randomize",
				"randomize",
				function () {
					seedWidget.value = -1;
				},
				{
					serialize: false,
				}
			);

			node.addWidget(
				"button",
				"🎲new fixed seed",
				"fixed",
				function () {
					// going to run into JavaScript number limitations
					// use API instead? (ugh)
					seedWidget.value = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
				},
				{
					serialize: false,
				}
			);

			const histWidget = node.addWidget(
				"button",
				"♻️previous seed",
				"previous",
				function () {
					seedWidget.value = this.value;
				},
				{
					serialize: false,
				}
			);
			histWidget.disabled = true;
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
			}

			if (node?.comfyClass === "PrivateSeed") {
				const histWidget = node.widgets.find(
					(w) => w.name === "♻️previous seed"
				);
				const values = event.detail.output.seed_value;
				histWidget.value = values[values.length - 1];
				histWidget.label = `♻️${histWidget.value}`;
				histWidget.disabled = false;
			}
		});
	},
});
