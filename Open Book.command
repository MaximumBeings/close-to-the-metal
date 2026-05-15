#!/bin/bash
# Double-click this file in Finder to open the book in your browser.
# It starts a local web server so all CSS, JS, and navigation work correctly.

cd "$(dirname "$0")/site"

# Kill any previous server on port 8765
lsof -ti:8765 | xargs kill -9 2>/dev/null

# Start Python web server in background
python3 -m http.server 8765 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 1

# Open in default browser
open http://localhost:8765

echo ""
echo "Book server running at http://localhost:8765"
echo "Press Ctrl+C to stop."

# Keep running until user presses Ctrl+C
trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'" EXIT
wait $SERVER_PID
