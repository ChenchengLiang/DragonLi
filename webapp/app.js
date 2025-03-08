const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const port = 3000;
const votesFile = path.join(__dirname, 'votes.json');

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/thesis', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'thesis.html'));
});

app.get('/vote', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'vote.html'));
});

app.use('/pdfs', express.static(path.join(__dirname, 'public', 'pdfs')));

// Load or initialize votes
function loadVotes() {
    try {
        if (fs.existsSync(votesFile)) {
            return JSON.parse(fs.readFileSync(votesFile, 'utf8'));
        }
    } catch (error) {
        console.error("Error reading votes.json:", error);
    }
    return [
        //{ id: 1, url: 'https://cdn.midjourney.com/a62a418d-e052-42b7-acaf-06c19f5d8157/0_3.png', votes: 0 },
        { id: 2, url: 'https://cdn.midjourney.com/fbbb0056-2a11-4906-9889-c614461ce2fc/0_1.png', votes: 0 },
        { id: 3, url: 'https://cdn.midjourney.com/de86765b-3f72-4a48-aadb-718442557d74/0_2.png', votes: 0 },
        { id: 4, url: 'https://cdn.midjourney.com/476f5e9d-317a-4e60-a2c6-21d9e38d266a/0_1.png', votes: 0 },
        { id: 5, url: 'https://cdn.midjourney.com/4ce6290e-6b9f-4ab5-8826-6b1ffdd12698/0_3.png', votes: 0 },
        { id: 6, url: 'https://cdn.midjourney.com/83d2c0d8-906f-4953-a1fd-6f64c43fca4b/0_2.png', votes: 0 },
        { id: 7, url: 'https://cdn.midjourney.com/09e53c3b-884f-44ea-a835-789ff0e48292/0_1.png', votes: 0 },
        { id: 8, url: 'https://cdn.midjourney.com/6715e58a-eb1f-43f3-a99a-9f3fda3f750b/0_2.png', votes: 0 },
        { id: 9, url: 'https://cdn.midjourney.com/e914ef1b-80c6-4741-911d-549d7c30ecd9/0_2.png', votes: 0 },
        { id: 10, url: 'https://cdn.midjourney.com/a57fc5e8-5c5e-4f82-9898-5b2c96fd3226/0_3.png', votes: 0 },
        { id: 11, url: 'https://cdn.midjourney.com/a57fc5e8-5c5e-4f82-9898-5b2c96fd3226/0_2.png', votes: 0 },
        { id: 12, url: 'https://cdn.midjourney.com/12b3c3d8-fd25-40ea-a4b1-ab2254c183ee/0_0.png', votes: 0 },
        { id: 13, url: 'https://cdn.midjourney.com/14369add-2434-4967-be54-7825361adbe3/0_3.png', votes: 0 },
        { id: 14, url: 'https://cdn.midjourney.com/14369add-2434-4967-be54-7825361adbe3/0_2.png', votes: 0 },
        { id: 15, url: 'https://cdn.midjourney.com/14369add-2434-4967-be54-7825361adbe3/0_1.png', votes: 0 },
        { id: 16, url: 'https://cdn.midjourney.com/14369add-2434-4967-be54-7825361adbe3/0_0.png', votes: 0 },
        { id: 17, url: 'https://cdn.midjourney.com/60357091-2a54-4e32-9354-f8e775d05ebe/0_2.png', votes: 0 },
        { id: 18, url: 'https://cdn.midjourney.com/955a1739-c71c-4472-9004-5a51e7602834/0_0.png', votes: 0 },
        { id: 20, url: 'https://cdn.midjourney.com/8d40b6aa-8bc5-46c0-8c0a-5b7164097590/0_1.png', votes: 0 },
        { id: 21, url: 'https://cdn.midjourney.com/8d40b6aa-8bc5-46c0-8c0a-5b7164097590/0_0.png', votes: 0 },
        { id: 22, url: 'https://cdn.midjourney.com/a3bfcf37-6e47-4907-9b5b-aa761c34750d/0_0.png', votes: 0 },
        { id: 23, url: 'https://cdn.midjourney.com/62d0df0c-7e75-4b84-8fb1-76e3989b24aa/0_2.png', votes: 0 },
        { id: 24, url: 'https://cdn.midjourney.com/62d0df0c-7e75-4b84-8fb1-76e3989b24aa/0_3.png', votes: 0 },
        { id: 25, url: 'https://cdn.midjourney.com/71ef9d05-9d57-4c31-8a28-001379c89ad6/0_0.png', votes: 0 },
        { id: 26, url: 'https://cdn.midjourney.com/48e0accf-dffa-4d76-8c22-6b7cb42077ec/0_1.png', votes: 0 },
        { id: 27, url: 'https://cdn.midjourney.com/bb654ca7-e759-4b84-aaf8-86a34dc18f4f/0_0.png', votes: 0 },
        { id: 28, url: 'https://cdn.midjourney.com/eba2031c-c018-4379-a2c6-68e025c384b1/0_1.png', votes: 0 },
        { id: 29, url: 'https://cdn.midjourney.com/eba2031c-c018-4379-a2c6-68e025c384b1/0_0.png', votes: 0 },
        { id: 30, url: 'https://cdn.midjourney.com/b33925fb-5dd8-4e05-a718-ef1cbcbc90d8/0_0.png', votes: 0 },
        { id: 31, url: 'https://cdn.midjourney.com/11435361-0530-4787-8dfa-0da02b4f3b91/0_3.png', votes: 0 },
        { id: 32, url: 'https://cdn.midjourney.com/3d742119-2fa5-4f8e-b083-7eb11c278879/0_2.png', votes: 0 },
        { id: 33, url: 'https://cdn.midjourney.com/bc9adfc4-2612-40a5-a7f7-49492c88a7ae/0_3.png', votes: 0 },
        { id: 34, url: 'https://cdn.midjourney.com/d1cb83bc-a62a-4339-b6ef-271631bcc118/0_0.png', votes: 0 },
        { id: 35, url: 'https://cdn.midjourney.com/d1cb83bc-a62a-4339-b6ef-271631bcc118/0_3.png', votes: 0 },
        { id: 36, url: 'https://cdn.midjourney.com/7ae283f4-1352-406a-97f7-f11333000693/0_2.png', votes: 0 },
        { id: 37, url: 'https://cdn.midjourney.com/f26b110d-8781-41d3-98a2-f4e4dcdb410d/0_0.png', votes: 0 },
        { id: 38, url: 'https://cdn.midjourney.com/c80e88a7-72bf-4f31-ba64-17be3d18cd1b/0_2.png', votes: 0 },
        { id: 39, url: 'https://cdn.midjourney.com/19bc3e73-3ba8-405e-935d-26f5b3fa65d8/0_1.png', votes: 0 },
        { id: 40, url: 'https://cdn.midjourney.com/9374b798-1f54-421a-af59-7846b1cf3df4/0_0.png', votes: 0 },
        { id: 41, url: 'https://cdn.midjourney.com/895e544a-d153-48e8-bcfc-9b77f0537bb9/0_0.png', votes: 0 },
        { id: 42, url: 'https://cdn.midjourney.com/a4ee29f7-ef18-416e-b2d3-bec7470d90b4/0_1.png', votes: 0 },
        { id: 43, url: 'https://cdn.midjourney.com/a4ee29f7-ef18-416e-b2d3-bec7470d90b4/0_0.png', votes: 0 },
        { id: 44, url: 'https://cdn.midjourney.com/3f92acfb-4671-4ea6-a354-dd3f950f8561/0_2.png', votes: 0 },
        { id: 45, url: 'https://cdn.midjourney.com/3f92acfb-4671-4ea6-a354-dd3f950f8561/0_0.png', votes: 0 },
        { id: 46, url: 'https://cdn.midjourney.com/cce5a2f2-c5c9-4058-ab2c-64324710fabb/0_1.png', votes: 0 },
        { id: 47, url: 'https://cdn.midjourney.com/461249dc-355f-4fa4-8cb5-ff7d7c1f1a9a/0_1.png', votes: 0 },
        { id: 48, url: 'https://cdn.midjourney.com/461249dc-355f-4fa4-8cb5-ff7d7c1f1a9a/0_2.png', votes: 0 },
        { id: 49, url: 'https://cdn.midjourney.com/bf25943c-0c7d-4356-a563-bb9784167b10/0_0.png', votes: 0 },
        { id: 50, url: 'https://cdn.midjourney.com/bf25943c-0c7d-4356-a563-bb9784167b10/0_1.png', votes: 0 },
        { id: 51, url: 'https://cdn.midjourney.com/bf25943c-0c7d-4356-a563-bb9784167b10/0_2.png', votes: 0 },
        { id: 52, url: 'https://cdn.midjourney.com/71bcb9d5-598b-4c00-ac8c-a3ffdcfb9dee/0_3.png', votes: 0 },
        { id: 53, url: 'https://cdn.midjourney.com/098e1afc-728e-4580-81ce-64e2e432ed39/0_0.png', votes: 0 },
        { id: 54, url: 'https://cdn.midjourney.com/098e1afc-728e-4580-81ce-64e2e432ed39/0_1.png', votes: 0 },
        { id: 55, url: 'https://cdn.midjourney.com/098e1afc-728e-4580-81ce-64e2e432ed39/0_2.png', votes: 0 },
        { id: 56, url: 'https://cdn.midjourney.com/098e1afc-728e-4580-81ce-64e2e432ed39/0_3.png', votes: 0 },
        { id: 57, url: 'https://cdn.midjourney.com/b7fabf09-b9b0-4144-b074-504c99ae63b0/0_1.png', votes: 0 },
        { id: 58, url: 'https://cdn.midjourney.com/b7fabf09-b9b0-4144-b074-504c99ae63b0/0_2.png', votes: 0 },
        { id: 59, url: 'https://cdn.midjourney.com/b7fabf09-b9b0-4144-b074-504c99ae63b0/0_3.png', votes: 0 },
        { id: 60, url: 'https://cdn.midjourney.com/7e81197a-8f67-43c4-922c-ecf1a8cfa59e/0_0.png', votes: 0 },
        { id: 61, url: 'https://cdn.midjourney.com/4eb39d9e-306c-41af-85fb-4b4e0bdf10f9/0_3.png', votes: 0 },
        { id: 62, url: 'https://cdn.midjourney.com/00daae3c-9f2e-4153-b803-66a680a026b5/0_1.png', votes: 0 },
        { id: 63, url: 'https://cdn.midjourney.com/00daae3c-9f2e-4153-b803-66a680a026b5/0_3.png', votes: 0 },
        { id: 64, url: 'https://cdn.midjourney.com/00daae3c-9f2e-4153-b803-66a680a026b5/0_0.png', votes: 0 },
        { id: 65, url: 'https://cdn.midjourney.com/e7b68ae7-6ed2-4620-ae2a-137c28cd2c11/0_2.png', votes: 0 },
        { id: 66, url: 'https://cdn.midjourney.com/827c4d51-e4aa-4f37-a0ba-117ba6f6a17b/0_1.png', votes: 0 },
        { id: 67, url: 'https://cdn.midjourney.com/2a497cdf-e62f-4938-a580-de56c5f1e8a5/0_3.png', votes: 0 },
        { id: 68, url: 'https://cdn.midjourney.com/5c5fee0f-a3e6-4b93-bb82-eb00ebc15136/0_0.png', votes: 0 },
        { id: 69, url: 'https://cdn.midjourney.com/201caeab-b96a-4572-9db1-2df4cae59af1/0_3.png', votes: 0 },
        { id: 70, url: 'https://cdn.midjourney.com/cc3a9e66-d171-4016-96c7-3506dc59b935/0_2.png', votes: 0 },
        { id: 71, url: 'https://cdn.midjourney.com/d52f3ee6-38ea-43c0-8b65-e01e905749be/0_0.png', votes: 0 },
        { id: 72, url: 'https://cdn.midjourney.com/f40f8d25-57c2-4e4b-9cf2-323a57f77807/0_3.png', votes: 0 },
        { id: 73, url: 'https://cdn.midjourney.com/cf101f33-f271-45e7-b9ee-b8cfa56c4855/0_0.png', votes: 0 },
        { id: 74, url: 'https://cdn.midjourney.com/c1efc8ed-3f04-48bf-a43a-d4a87f8affff/0_1.png', votes: 0 },
        { id: 75, url: 'https://cdn.midjourney.com/d71a6f27-c59d-4113-af00-b37eab503f7f/0_0.png', votes: 0 },
        { id: 76, url: 'https://cdn.midjourney.com/5fdcfbda-7d2f-4ce4-babf-547a768e58f0/0_2.png', votes: 0 },
        { id: 77, url: 'https://cdn.midjourney.com/5fdcfbda-7d2f-4ce4-babf-547a768e58f0/0_0.png', votes: 0 },
        { id: 78, url: 'https://cdn.midjourney.com/5597ad04-3058-43db-985c-6be0633fac2a/0_0.png', votes: 0 },
        { id: 79, url: 'https://cdn.midjourney.com/bd2cb93e-db08-4600-a328-5932daea564b/0_3.png', votes: 0 },
        { id: 80, url: 'https://cdn.midjourney.com/7deb607f-2cbf-4090-a8f7-ec0d2f2a3139/0_3.png', votes: 0 },
        { id: 81, url: 'https://cdn.midjourney.com/7deb607f-2cbf-4090-a8f7-ec0d2f2a3139/0_2.png', votes: 0 },
        { id: 82, url: 'https://cdn.midjourney.com/7deb607f-2cbf-4090-a8f7-ec0d2f2a3139/0_1.png', votes: 0 },
        { id: 83, url: 'https://cdn.midjourney.com/7deb607f-2cbf-4090-a8f7-ec0d2f2a3139/0_0.png', votes: 0 },
        { id: 84, url: 'https://cdn.midjourney.com/62a48786-79f0-436d-85a1-3f3ff2052a4e/0_0.png', votes: 0 },
        { id: 85, url: 'https://cdn.midjourney.com/ec275991-c60c-4cbb-94f3-e49eff43d41f/0_1.png', votes: 0 },
        { id: 86, url: 'https://cdn.midjourney.com/254203e9-8dfc-4750-a55c-4dabf81f34db/0_1.png', votes: 0 },
        { id: 87, url: 'https://cdn.midjourney.com/9fcf01e4-dfe8-43d0-b644-7e7964e76e04/0_0.png', votes: 0 },
        { id: 88, url: 'https://cdn.midjourney.com/8970e2ce-c8fb-4053-a6f3-2783acd7027d/0_3.png', votes: 0 },
        { id: 89, url: 'https://cdn.midjourney.com/90888ac0-0ded-4aa6-97dc-5f84d25f02a4/0_0.png', votes: 0 },
        //{ id: 90, url: '', votes: 0 },
        //{ id: 91, url: '', votes: 0 },
        //{ id: 92, url: '', votes: 0 },
        //{ id: 93, url: '', votes: 0 },
        { id: 94, url: '', votes: 0 }
    ];
}

function saveVotes(pictures) {
    try {
        fs.writeFileSync(votesFile, JSON.stringify(pictures, null, 2), 'utf8');
    } catch (error) {
        console.error("Error saving votes.json:", error);
    }
}

let pictures = loadVotes();

app.get('/vote-data', (req, res) => {
    res.json(pictures);
});

app.post('/vote', (req, res) => {
    const { id } = req.body;
    const picture = pictures.find(p => p.id === id);
    if (picture) {
        picture.votes += 1;
        saveVotes(pictures);
        res.json({ success: true, pictures });
    } else {
        res.status(400).json({ success: false, message: 'Invalid picture ID' });
    }
});

// Reset voting counts and save to JSON file
app.post('/reset-votes', (req, res) => {
    pictures.forEach(picture => picture.votes = 0);
    saveVotes(pictures);
    res.json({ success: true, message: "All votes have been reset.", pictures });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
