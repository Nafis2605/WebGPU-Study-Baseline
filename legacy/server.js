const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();
const PORT = 5500;

// Middleware to handle CORS issues
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*'); // Allow all origins
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  
  // Handle preflight requests for OPTIONS method
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  next();
});

// Proxy setup to bypass CORS issues for tfhub.dev
app.use('/tfhub', createProxyMiddleware({
  target: 'https://tfhub.dev',
  changeOrigin: true,
  pathRewrite: {
    '^/tfhub': '', // Removes the /tfhub prefix
  },
  onProxyRes: function(proxyRes, req, res) {
    proxyRes.headers['Access-Control-Allow-Origin'] = '*'; // Add CORS headers to the response
  },
}));

// Serve static files from the current directory
app.use(express.static(path.join(__dirname, '/')));

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
  console.log(`Test cross-origin isolation at http://localhost:${PORT}/your-page.html`);
});
