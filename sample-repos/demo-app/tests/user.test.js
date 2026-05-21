const { registerUser } = require("../src/controllers/userController");

describe("user registration", () => {
  test("returns a validation error when email is missing", () => {
    const req = {
      body: {
        name: "Ada Lovelace",
        password: "secret-password",
      },
    };
    const res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn(),
    };

    registerUser(req, res);

    expect(res.status).toHaveBeenCalledWith(400);
    expect(res.json).toHaveBeenCalledWith({
      error: "Name, email, and password are required for user registration.",
    });
  });

  test("returns a validation error when email is invalid", () => {
    const req = {
      body: {
        name: "Ada Lovelace",
        email: "invalid-email",
        password: "secret-password",
      },
    };
    const res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn(),
    };

    registerUser(req, res);

    expect(res.status).toHaveBeenCalledWith(400);
    expect(res.json).toHaveBeenCalledWith({
      error: "Invalid email address.",
    });
  });
});