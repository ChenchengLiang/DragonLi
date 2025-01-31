const express = require('express');
const path = require('path');
const fs = require('fs');

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

// ================== MODIFY THIS ROUTE ==================
app.post('/process', (req, res) => {
  let userInput = req.body.userInput || "";

  // Trim trailing whitespace
  userInput = userInput.trimEnd();

  // Check if userInput ends with "SatGlucose(100)"
  if (!userInput.endsWith("SatGlucose(100)")) {
    // If not, add a new line plus "SatGlucose(100)"
    userInput += "\nSatGlucose(100)";
  }

  try {
    // Write the user input to user-input.eq
    fs.writeFileSync("user-input.eq", userInput, "utf8");

    // Respond with a success message
    res.json({
      message: `Successfully stored input to user-input.eq.\nFinal text:\n${userInput}`
    });
  } catch (error) {
    console.error("Error writing to user-input.eq:", error);
    res.status(500).json({ message: "Error saving input to file." });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
