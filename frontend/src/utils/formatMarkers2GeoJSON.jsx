function formatData(markers, selectedSubstation, allowEmpty) {
    const validMarkers = markers[selectedSubstation.SS_ID].filter((marker) => {
        if (typeof marker.getLabel === "function") {
            const label = marker.getLabel().text;
            return typeof label === "string" && label.trim() !== "";
        } else {
            const label = marker.label.text;
            return typeof label === "string" && label.trim() !== "";
        }
    });

    if (
        (!markers[selectedSubstation.SS_ID] ||
            markers[selectedSubstation.SS_ID].length === 0 ||
            validMarkers.length === 0) &&
        !allowEmpty
    ) {
        console.error("Markers list is empty or contains invalid labels.");
        return;
    }

    const markersToSave = allowEmpty
        ? markers[selectedSubstation.SS_ID]
        : validMarkers;

    const geojson = {
        type: "FeatureCollection",
        features: [
            {
                type: "Feature",
                geometry: {
                    type: "MultiPoint",
                    coordinates: markersToSave.map((marker) => [
                        marker.getPosition().lng(),
                        marker.getPosition().lat(),
                    ]),
                },
                properties: {
                    SS_ID: selectedSubstation.SS_ID,
                    SS_NAME: selectedSubstation.SS_NAME,
                    SS_OPERATOR: selectedSubstation.SS_OPERATOR,
                    SS_TYPE: selectedSubstation.SS_TYPE,
                    SS_VOLTAGE: selectedSubstation.SS_VOLTAGE,
                    connected_tl_id: selectedSubstation.connected_tl_id,
                    LINE_VOLTS: selectedSubstation.LINE_VOLTS,
                    REGION: selectedSubstation.REGION,
                    REGION_ID: selectedSubstation.REGION_ID,
                    threePhaseTransformerCount: 0,
                    singlePhaseTransformerCount: 0,
                    primaryPowerLineCount: 0,
                    secondaryPowerLineCount: 0,
                    totalTransformerCount: 0,
                    markers: markersToSave.map((marker) => {
                        const labelObj = marker.getLabel();
                        const labelKey = labelObj.text;
                        const attributes = marker.attributes;
                        return {
                            label: labelKey,
                            attributes: attributes,
                            color: labelObj.color,
                        };
                    }),
                },
            },
        ],
    };

    return {geojson, markersToSave};
}


export default formatData;