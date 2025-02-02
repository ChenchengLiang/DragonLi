// app.js
const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const port = 3000;

// Middleware to parse JSON and URL-encoded form data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from the "public" directory (including index.html, etc.)
app.use(express.static(path.join(__dirname, 'public')));

// GET /load-eq: read example.eq and return its content
app.get('/load-eq', (req, res) => {
  try {
    const data = fs.readFileSync('example.eq', 'utf8');
    res.send(data);
  } catch (err) {
    console.error(err);
    res.status(500).send('Error reading the .eq file.');
  }
});

// POST /process: write user input, run shell script, read solver output
app.post('/process', (req, res) => {
  // Extract data from the request body
  let userInput = req.body.userInput || "";
  const solverType = req.body.solverType || "";
  console.log("Received solverType:", solverType);

  const rankTask = req.body.rankTask || "";
  console.log("Received rankTask:", rankTask);

  const benchmark = req.body.benchmark || "";
  console.log("Received benchmark:", benchmark);

  // Trim trailing whitespace
  userInput = userInput.trimEnd();

  // Ensure userInput ends with "SatGlucose(100)"
  if (!userInput.endsWith("SatGlucose(100)")) {
    userInput += "\nSatGlucose(100)";
  }

  // 1) Write the user input to "user-input.eq"
  try {
    fs.writeFileSync("user-input.eq", userInput, "utf8");
  } catch (error) {
    console.error("Error writing to user-input.eq:", error);
    return res.status(500).json({ message: "Error saving input to file." });
  }

  // 2) Run the shell script, passing solverType as an argument
  //    If your script is bash-based, "sh" is fine. If it’s a different shell or
  //    you have it as executable with a shebang, you might do:
  //    spawn('./run_solver.sh', [solverType])
  const solverProcess = spawn('sh', ['run_solver.sh', solverType, rankTask, benchmark]);

  let stderrData = '';

  solverProcess.stderr.on('data', (data) => {
    stderrData += data.toString();
  });

  solverProcess.on('close', (code) => {
    // Handle a non-zero exit code (script error or something else)
    if (code !== 0) {
      console.error(`Solver script exited with code ${code}`);
      console.error(`Error output: ${stderrData}`);
      return res
        .status(500)
        .json({ message: 'Error running solver script.\n' + stderrData });
    }

    // 3) If the script was successful, read the output file (answer.txt)
    try {
      const answer = fs.readFileSync('answer.txt', 'utf8');
      // 4) Return the solver result
      res.json({
        message: 'Solver finished successfully.',
        solverOutput: answer
      });
    } catch (err) {
      console.error("Error reading answer.txt:", err);
      //res.status(500).json({ message: "Could not read answer.txt" });
      res.status(500).json({ message: "Wrong model" });
    }
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
