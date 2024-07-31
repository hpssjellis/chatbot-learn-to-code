


function fitCurve() {
    const xData = document.getElementById('myXData').value.split(',').map(Number);
    const yData = document.getElementById('myYData').value.split(',').map(Number);
    const initialGuess = [1, 1, 0, 0];
    const [A, w, b, C] = gradientDescent(xData, yData, initialGuess, sinusoidal);

    console.log(`A: ${A}, w: ${w}, b: ${b}, C: ${C}`);

    // Generate y values for the fitted curve
    const fittedYData = xData.map(x => sinusoidal(x, A, w, b, C));

    // Plot the original data and fitted curve
    const originalTrace = {
        x: xData,
        y: yData,
        mode: 'markers',
        type: 'scatter',
        name: 'Original Data'
    };

    const fittedTrace = {
        x: xData,
        y: fittedYData,
        mode: 'lines',
        type: 'scatter',
        name: 'Fitted Curve'
    };

    const layout = {
        title: 'Sinusoidal Curve Fitting',
        xaxis: { title: 'X (Degrees)' },
        yaxis: { title: 'Y' }
    };

    Plotly.newPlot('myPlot', [originalTrace, fittedTrace], layout);
}
