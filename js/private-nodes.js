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
		});
	},
});
