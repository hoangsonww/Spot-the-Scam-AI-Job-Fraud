#!/bin/bash

# Spot the Scam - Wiki Server Launcher
# This script starts a simple HTTP server to view the wiki

PORT=${1:-8080}
WIKI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🛡️  Spot the Scam - Wiki Server"
echo "================================"
echo ""
echo "📂 Serving from: $WIKI_DIR"
echo "🌐 Port: $PORT"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "🐍 Using Python 3..."
    echo "🚀 Starting server at http://localhost:$PORT"
    echo "📖 Press Ctrl+C to stop the server"
    echo ""
    cd "$WIKI_DIR" && python3 -m http.server $PORT
elif command -v python &> /dev/null; then
    echo "🐍 Using Python..."
    echo "🚀 Starting server at http://localhost:$PORT"
    echo "📖 Press Ctrl+C to stop the server"
    echo ""
    cd "$WIKI_DIR" && python -m http.server $PORT
# Check if Node.js is available
elif command -v npx &> /dev/null; then
    echo "📦 Using Node.js http-server..."
    echo "🚀 Starting server at http://localhost:$PORT"
    echo "📖 Press Ctrl+C to stop the server"
    echo ""
    cd "$WIKI_DIR" && npx http-server . -p $PORT
# Check if PHP is available
elif command -v php &> /dev/null; then
    echo "🐘 Using PHP..."
    echo "🚀 Starting server at http://localhost:$PORT"
    echo "📖 Press Ctrl+C to stop the server"
    echo ""
    cd "$WIKI_DIR" && php -S localhost:$PORT
else
    echo "❌ Error: No suitable server found!"
    echo ""
    echo "Please install one of the following:"
    echo "  - Python 3: https://www.python.org/"
    echo "  - Node.js: https://nodejs.org/"
    echo "  - PHP: https://www.php.net/"
    echo ""
    echo "Or simply open index.html directly in your browser:"
    echo "  open index.html (macOS)"
    echo "  xdg-open index.html (Linux)"
    echo "  start index.html (Windows)"
    exit 1
fi
