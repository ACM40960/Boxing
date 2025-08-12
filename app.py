
from flask import Flask, render_template, Response
from scripts import two_player_analyzer
from scripts import shadow_boxing_analyzer 

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/two_player')
def two_player():
    """Renders the two-player fight page."""
    return render_template('two_player.html')
    

@app.route('/shadow_boxing')
def shadow_boxing():
    """Renders the shadow boxing page."""
    return render_template('shadow_boxing.html')

@app.route('/video_feed_two_player')
def video_feed_two_player():
    """Provides the video stream for the two-player analyzer."""
    return Response(two_player_analyzer.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_shadow_boxing')
def video_feed_shadow_boxing():
    """Provides the video stream for the shadow boxing trainer."""
    return Response(shadow_boxing_analyzer.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
