var dagfuncs = window.dash_ag_grid = window.dash_ag_grid || {};

dagfuncs.Median = function (params) {
    if (!params.values || params.values.length === 0) return null;
    var values = params.values.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(values.length / 2);
    // Handle both even and odd length
    return values.length % 2 !== 0 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
};

dagfuncs.Mode = function (params) {
    if (!params.values || params.values.length === 0) return null;
    var frequency = {};
    var maxFreq = 0;
    var mode = params.values[0];
    for (var i = 0; i < params.values.length; i++) {
        var v = params.values[i];
        if (v !== null && v !== undefined) {
            frequency[v] = (frequency[v] || 0) + 1;
            if (frequency[v] > maxFreq) {
                maxFreq = frequency[v];
                mode = v;
            }
        }
    }
    return mode;
};

dagfuncs.StdDev = function (params) {
    if (!params.values || params.values.length === 0) return null;
    var n = params.values.length;
    var validValues = params.values.filter(v => v !== null && !isNaN(v));
    if (validValues.length === 0) return null;
    n = validValues.length;

    var mean = validValues.reduce(function (a, b) { return a + b; }, 0) / n;
    var variance = validValues.reduce(function (a, b) { return a + Math.pow(b - mean, 2); }, 0) / n;
    return Math.sqrt(variance);
};

dagfuncs.WeightedAvg = function (params) {
    // Requires access to a weight column.
    // Configuration: params.colDef.weightCol (specific) or params.context.weightCol (global default).

    var weightCol = 'Quantity'; // Default fallback

    // Check specific column configuration first
    if (params.colDef && params.colDef.weightCol) {
        weightCol = params.colDef.weightCol;
    }
    // Fallback to global context if available
    else if (params.context && params.context.weightCol) {
        weightCol = params.context.weightCol;
    }

    // params.rowNode.allLeafChildren gives access to all rows in this group
    if (!params.rowNode || !params.rowNode.allLeafChildren) return null;

    var nodes = params.rowNode.allLeafChildren;
    var totalWeight = 0;
    var weightedSum = 0;
    var valCol = params.colDef.field;

    for (var i = 0; i < nodes.length; i++) {
        var d = nodes[i].data;
        if (d) {
            var val = parseFloat(d[valCol]);
            var w = parseFloat(d[weightCol]);
            if (!isNaN(val) && !isNaN(w)) {
                weightedSum += val * w;
                totalWeight += w;
            }
        }
    }
    return totalWeight !== 0 ? weightedSum / totalWeight : null;
};
