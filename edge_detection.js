let edgeLayerVisible = false;
function toggleEdgeLayer() {
        edgeLayerVisible = !edgeLayerVisible;
        viewer.world.getItemAt(1).setOpacity(edgeLayerVisible ? 1 : 0);
    }