function registerUser(req, res) {
  const { name, email, password } = req.body;

  // Registration validation currently checks only required user fields.
  if (!name || !email || !password) {
    return res.status(400).json({
      error: "Name, email, and password are required for user registration.",
    });
  }

  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({
      error: "Invalid email address.",
    });
  }

  return res.status(201).json({
    message: "User registration successful.",
    user: {
      name,
      email,
    },
  });
}

module.exports = {
  registerUser,
};