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

// 1) Change the /load-eq route to read from "user-input.eq" by default:
app.get('/load-eq', (req, res) => {
  try {
    // First try reading user-input.eq
    const data = fs.readFileSync('user-input.eq', 'utf8');
    res.send(data);
  } catch (err) {
    console.error('Could not read user-input.eq. Falling back to example.eq:', err);
    // If "user-input.eq" doesn’t exist or can’t be read, optionally fall back to example.eq
    try {
      const fallbackData = fs.readFileSync('example.eq', 'utf8');
      res.send(fallbackData);
    } catch (fallbackErr) {
      console.error('Could not read example.eq either:', fallbackErr);
      res.status(500).send('Error reading input files.');
    }
  }
});

// POST /process: write user input, run solver, read solver output
app.post('/process', (req, res) => {
  let userInput = req.body.userInput || "";
  const solverType = req.body.solverType || "";
  console.log("Received solverType:", solverType);

  const rankTask = req.body.rankTask || "";
  console.log("Received rankTask:", rankTask);

  const benchmark = req.body.benchmark || "";
  console.log("Received benchmark:", benchmark);

  const timeout_in_second = req.body.timeout_in_second || "";
  console.log("Received timeout_in_second:", timeout_in_second);

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

  // 2) Run the shell script, passing solverType, rankTask, benchmark, etc.
  const solverProcess = spawn('sh', ['run_solver.sh', solverType, rankTask, benchmark, timeout_in_second]);

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
      res.status(500).json({ message: "Syntax error" });
    }
  });
});

// POST /generate-benchmark
app.post('/generate-benchmark', (req, res) => {
  const benchmark = req.body.benchmark || "";
  console.log("Generating benchmark for:", benchmark);

  const generatorProcess = spawn('sh', ['run_generator.sh', benchmark]);
  let stderrData = '';

  generatorProcess.stderr.on('data', (data) => {
    stderrData += data.toString();
  });

  generatorProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Generator script exited with code ${code}`);
      console.error(`Error output: ${stderrData}`);
      return res.status(500).json({ message: 'Error running generator script.\n' + stderrData });
    }

    // Successfully generated user-input.eq
    res.status(200).json({ message: 'Benchmark generated successfully.' });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
