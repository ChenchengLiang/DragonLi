const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');  // <-- Import spawn

const app = express();
const port = 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files (including your index.html)
app.use(express.static(path.join(__dirname, 'public')));

// Endpoint to read and return the content of example.eq (unchanged)
app.get('/load-eq', (req, res) => {
  try {
    const data = fs.readFileSync('example.eq', 'utf8');
    res.send(data);
  } catch (err) {
    console.error(err);
    res.status(500).send('Error reading the .eq file.');
  }
});

// POST /process route
app.post('/process', (req, res) => {
  let userInput = req.body.userInput || "";

  // Trim trailing whitespace
  userInput = userInput.trimEnd();

  // Ensure ends with "SatGlucose(100)"
  if (!userInput.endsWith("SatGlucose(100)")) {
    userInput += "\nSatGlucose(100)";
  }

  // 1) Write user input to user-input.eq
  try {
    fs.writeFileSync("user-input.eq", userInput, "utf8");
  } catch (error) {
    console.error("Error writing to user-input.eq:", error);
    return res.status(500).json({ message: "Error saving input to file." });
  }

  // 2) Run shell script (sh run_solver.sh)
  const solverProcess = spawn('sh', ['run_solver.sh']);

  let stderrData = '';

  solverProcess.stderr.on('data', (data) => {
    stderrData += data.toString();
  });

  solverProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Solver script exited with code ${code}`);
      console.error(`Error output: ${stderrData}`);
      return res
        .status(500)
        .json({ message: 'Error running solver script.\n' + stderrData });
    }

    // 3) Read answer.txt after the solver finishes
    try {
      const answer = fs.readFileSync('answer.txt', 'utf8');
      // 4) Send answer back to client
      res.json({
        message: 'Solver finished successfully.',
        solverOutput: answer
      });
    } catch (err) {
      console.error("Error reading answer.txt:", err);
      res.status(500).json({ message: "Could not read answer.txt" });
    }
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
