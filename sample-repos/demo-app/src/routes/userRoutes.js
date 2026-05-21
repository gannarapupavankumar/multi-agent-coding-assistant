const express = require("express");
const { registerUser } = require("../controllers/userController");

const router = express.Router();

// User registration route handled by the user controller.
router.post("/register", registerUser);

module.exports = router;
