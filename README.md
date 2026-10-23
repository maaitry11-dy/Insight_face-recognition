# \# 🎥 InsightFace Face Recognition \& Attendance Tracking

# 

# A real-time face recognition and person tracking system built with \*\*InsightFace\*\*, \*\*Deep SORT\*\*, \*\*OpenCV\*\*, \*\*Flask\*\*, and \*\*MySQL\*\*.

# 

# The application processes a video stream, detects faces using InsightFace, tracks individuals using Deep SORT, matches detected faces against a collection of known faces, saves detection snapshots, and records attendance/tracking events in a MySQL database.

# 

# \## 👤 GitHub

# 

# \*\*GitHub:\*\* \[maaitry11-dy](https://github.com/maaitry11-dy)

# 

# \---

# 

# \## ✨ Features

# 

# \- 🔍 Face detection using InsightFace

# \- 🧠 Face recognition using face embeddings

# \- 🚶 Person/face tracking using Deep SORT

# \- 🆔 Unique tracking IDs for detected individuals

# \- 📸 Automatically saves detected face images

# \- 👤 Supports known and unknown faces

# \- 🗄️ MySQL attendance tracking

# \- 🕐 Timestamp recording with timezone support

# \- 🌐 Flask-based live video stream

# \- 🎯 Configurable detection region

# \- ⏱️ Cooldown mechanism to avoid duplicate attendance records

# 

# \---

# 

# \## 🛠️ Technologies Used

# 

# | Technology | Purpose |

# |---|---|

# | Python | Main programming language |

# | OpenCV | Video processing and image manipulation |

# | InsightFace | Face detection and recognition |

# | Deep SORT | Object/face tracking |

# | NumPy | Numerical operations |

# | Flask | Web application and video streaming |

# | MySQL | Attendance/event storage |

# | mysql-connector-python | MySQL database connection |

# | FilterPy | Deep SORT dependency |

# | scikit-image | Image processing dependency |

# 

# \---

# 

# \## 📋 Requirements

# 

# \### Hardware

# 

# For GPU acceleration, the project is configured to use:

# 

# \- NVIDIA GPU

# \- CUDA-compatible environment

# \- Appropriate NVIDIA drivers

# \- CUDA/cuDNN compatible with your installed InsightFace/ONNX Runtime setup

# 

# The current code uses:

# 

# ```python

# providers=\['CUDAExecutionProvider']

# ```

# 

# If you do not have a CUDA-enabled GPU, you will need to change the provider to CPU.

# 

# \---

# 

# \## 📦 Installation

# 

# \### 1. Clone the repository

# 

# ```bash

# git clone https://github.com/maaitry11-dy/<YOUR-REPOSITORY-NAME>.git

# cd <YOUR-REPOSITORY-NAME>

# ```

# 

# \### 2. Create a virtual environment

# 

# ```bash

# python3 -m venv venv

# ```

# 

# Activate it on Linux/macOS:

# 

# ```bash

# source venv/bin/activate

# ```

# 

# On Windows:

# 

# ```cmd

# venv\\Scripts\\activate

# ```

# 

# \### 3. Upgrade pip

# 

# ```bash

# pip install --upgrade pip

# ```

# 

# \### 4. Install dependencies

# 

# ```bash

# pip install scikit-image lap filterpy

# pip install opencv-python opencv-contrib-python

# pip install insightface Flask numpy mysql-connector-python deep\_sort\_realtime

# ```

# 

# \---

# 

# \## 📁 Project Structure

# 

# A recommended project structure is:

# 

# ```text

# .

# ├── app.py

# ├── README.md

# ├── known\_faces/

# │   ├── person1.jpg

# │   ├── person2.jpg

# │   └── person3.jpg

# │

# ├── detected\_faces/

# │   ├── person1\_1\_20261023\_120000.jpg

# │   ├── Unknown\_2\_20261023\_120010.jpg

# │   └── ...

# │

# └── office\_videos/

# &#x20;   └── office\_leave\_1.mov

# ```

# 

# > Replace `app.py` with the actual filename containing your Python code.

# 

# \---

# 

# \# 👤 Known Faces

# 

# Create a `known\_faces` directory:

# 

# ```bash

# mkdir -p known\_faces

# ```

# 

# Place reference images of known people inside this directory.

# 

# For example:

# 

# ```text

# known\_faces/

# ├── John.jpg

# ├── Alice.jpg

# ├── David.jpg

# └── Smith.png

# ```

# 

# The filename is used as the person's name.

# 

# For example:

# 

# ```text

# John.jpg

# ```

# 

# will be loaded as:

# 

# ```text

# John

# ```

# 

# The application uses the first detected face from each reference image to generate the stored face embedding.

# 

# \### Recommended

# 

# Use:

# 

# \- Clear frontal face images

# \- Good lighting

# \- One person per image

# \- High-resolution images

# \- Minimal occlusion

# 

# \---

# 

# \# 🎥 Video Source

# 

# The current application uses a local video file:

# 

# ```python

# rtsp\_url="/home/ubuntu/office\_videos/office\_leave\_1.mov"

# ```

# 

# Despite the variable being named `rtsp\_url`, this is currently a local video file.

# 

# Change it to your video path:

# 

# ```python

# rtsp\_url="/path/to/your/video.mov"

# ```

# 

# You can also use an RTSP camera stream if supported by your OpenCV installation:

# 

# ```python

# rtsp\_url="rtsp://username:password@camera-ip:554/stream"

# ```

# 

# \---

# 

# \# 🎯 Detection Region

# 

# The application processes a selected region of the video:

# 

# ```python

# start\_x = 650

# start\_y = 350

# end\_x = 1200

# end\_y = 650

# ```

# 

# The region is cropped using:

# 

# ```python

# center\_frame = frame\[start\_y:end\_y, start\_x:end\_x]

# ```

# 

# Adjust these coordinates according to your camera/video resolution.

# 

# The detection region is displayed on the original frame using a rectangle.

# 

# \---

# 

# \# 🧠 Face Recognition

# 

# InsightFace is initialized using the `buffalo\_l` model:

# 

# ```python

# face\_app = FaceAnalysis(

# &#x20;   name='buffalo\_l',

# &#x20;   providers=\['CUDAExecutionProvider']

# )

# 

# face\_app.prepare(ctx\_id=0)

# ```

# 

# Known face embeddings are loaded during application startup.

# 

# The application calculates cosine similarity between the detected face embedding and known embeddings.

# 

# The current recognition threshold is:

# 

# ```python

# threshold=0.5

# ```

# 

# You can adjust this value:

# 

# ```python

# def match\_face(face\_embedding, threshold=0.5):

# ```

# 

# A higher threshold generally makes recognition more strict, while a lower threshold can increase matches but may also increase false positives.

# 

# \---

# 

# \# 🚶 Deep SORT Tracking

# 

# Deep SORT is initialized with:

# 

# ```python

# tracker = DeepSort(max\_age=30)

# ```

# 

# Each detected face is passed to Deep SORT:

# 

# ```python

# tracks = tracker.update\_tracks(

# &#x20;   detections,

# &#x20;   frame=center\_frame

# )

# ```

# 

# Each confirmed track receives a unique tracking ID:

# 

# ```python

# track\_id = track.track\_id

# ```

# 

# This tracking ID is used to identify the same person across multiple frames.

# 

# \---

# 

# \# ⏱️ Attendance Cooldown

# 

# To prevent the same person from generating a database record on every frame, the application uses a cooldown mechanism.

# 

# The current cooldown is:

# 

# ```python

# 60 seconds

# ```

# 

# A record is created only when:

# 

# ```python

# (now\_est - last\_seen\[track\_id]).total\_seconds() > 60

# ```

# 

# You can change the cooldown by modifying:

# 

# ```python

# > 60

# ```

# 

# For example, for a 5-minute cooldown:

# 

# ```python

# > 300

# ```

# 

# \---

# 

# \# 🗄️ MySQL Database

# 

# The application connects to MySQL using:

# 

# ```python

# db = mysql.connector.connect(

# &#x20;   host="localhost",

# &#x20;   user="root",

# &#x20;   password="password",

# &#x20;   database="fr"

# )

# ```

# 

# Before running the application, create the database:

# 

# ```sql

# CREATE DATABASE fr;

# ```

# 

# Then select it:

# 

# ```sql

# USE fr;

# ```

# 

# \---

# 

# \## 📊 Attendance Table

# 

# The code references an `attendance\_insight` table.

# 

# Create it with:

# 

# ```sql

# CREATE TABLE attendance\_insight (

# &#x20;   id INT AUTO\_INCREMENT PRIMARY KEY,

# &#x20;   name VARCHAR(255),

# &#x20;   type VARCHAR(50),

# &#x20;   timestamp DATETIME

# );

# ```

# 

# \---

# 

# \## 📊 Tracking Table

# 

# The application inserts records into:

# 

# ```text

# attendance\_insight\_tracking

# ```

# 

# Create the table using:

# 

# ```sql

# CREATE TABLE attendance\_insight\_tracking (

# &#x20;   id INT AUTO\_INCREMENT PRIMARY KEY,

# &#x20;   name VARCHAR(255),

# &#x20;   track\_id INT,

# &#x20;   type VARCHAR(50),

# &#x20;   timestamp DATETIME

# );

# ```

# 

# The application currently records:

# 

# ```text

# type = exit

# ```

# 

# Example:

# 

# ```text

# name       | track\_id | type | timestamp

# \-----------|----------|------|-------------------

# John       | 1        | exit | 2026-10-23 19:00:00

# Unknown    | 2        | exit | 2026-10-23 19:01:15

# Alice      | 3        | exit | 2026-10-23 19:02:30

# ```

# 

# \---

# 

# \# 🔐 Database Configuration

# 

# For security, avoid hardcoding your database password in production.

# 

# The current code contains:

# 

# ```python

# password="password"

# ```

# 

# A better approach is to use environment variables:

# 

# ```python

# db = mysql.connector.connect(

# &#x20;   host=os.getenv("DB\_HOST", "localhost"),

# &#x20;   user=os.getenv("DB\_USER", "root"),

# &#x20;   password=os.getenv("DB\_PASSWORD"),

# &#x20;   database=os.getenv("DB\_NAME", "fr")

# )

# ```

# 

# Then configure the environment variables before starting the application.

# 

# \---

# 

# \# 🕐 Timezone

# 

# The application gets the current UTC time and converts it to Eastern Time:

# 

# ```python

# now\_utc = datetime.now(ZoneInfo("UTC"))

# now\_est = now\_utc.astimezone(

# &#x20;   ZoneInfo("America/New\_York")

# )

# ```

# 

# If your deployment is in another timezone, change:

# 

# ```python

# ZoneInfo("America/New\_York")

# ```

# 

# For example:

# 

# ```python

# ZoneInfo("Asia/Kolkata")

# ```

# 

# \---

# 

# \# ▶️ Running the Application

# 

# Start the application:

# 

# ```bash

# python app.py

# ```

# 

# The Flask server will run on:

# 

# ```text

# http://0.0.0.0:5003

# ```

# 

# From the same machine, open:

# 

# ```text

# http://localhost:5003

# ```

# 

# The web page displays the video stream:

# 

# ```text

# 🎥 Live InsightFace Stream

# ```

# 

# \---

# 

# \# 🌐 Flask Endpoints

# 

# \## `/`

# 

# Displays the main web page.

# 

# ```text

# GET /

# ```

# 

# \## `/video\_feed`

# 

# Provides the live MJPEG video stream.

# 

# ```text

# GET /video\_feed

# ```

# 

# The browser receives frames using:

# 

# ```text

# multipart/x-mixed-replace

# ```

# 

# \---

# 

# \# 📸 Detected Face Images

# 

# When a person is detected after the cooldown period, a snapshot is saved in:

# 

# ```text

# detected\_faces/

# ```

# 

# Known faces are saved using:

# 

# ```text

# {name}\_{track\_id}\_{timestamp}.jpg

# ```

# 

# Example:

# 

# ```text

# John\_4\_20261023\_190000.jpg

# ```

# 

# Unknown faces are saved using:

# 

# ```text

# Unknown\_{track\_id}\_{timestamp}.jpg

# ```

# 

# Example:

# 

# ```text

# Unknown\_7\_20261023\_190010.jpg

# ```

# 

# \---

# 

# \# ⚙️ Important Configuration

# 

# The following values can be adjusted according to your environment:

# 

# \### Known faces directory

# 

# ```python

# KNOWN\_FACES\_DIR = '/home/ubuntu/fr\_simple/known\_faces'

# ```

# 

# \### Video source

# 

# ```python

# rtsp\_url="/home/ubuntu/office\_videos/office\_leave\_1.mov"

# ```

# 

# \### Detection region

# 

# ```python

# start\_x = 650

# start\_y = 350

# end\_x = 1200

# end\_y = 650

# ```

# 

# \### Recognition threshold

# 

# ```python

# threshold=0.5

# ```

# 

# \### Deep SORT maximum age

# 

# ```python

# tracker = DeepSort(max\_age=30)

# ```

# 

# \### Attendance cooldown

# 

# ```python

# > 60

# ```

# 

# \### Flask port

# 

# ```python

# app.run(

# &#x20;   host='0.0.0.0',

# &#x20;   port=5003

# )

# ```

# 

# \---

# 

# \# 🖥️ CPU-Only Configuration

# 

# The current project is configured for NVIDIA CUDA:

# 

# ```python

# providers=\['CUDAExecutionProvider']

# ```

# 

# If you don't have a CUDA-enabled GPU, use CPU execution instead:

# 

# ```python

# face\_app = FaceAnalysis(

# &#x20;   name='buffalo\_l',

# &#x20;   providers=\['CPUExecutionProvider']

# )

# 

# face\_app.prepare(ctx\_id=0)

# ```

# 

# GPU processing is recommended for real-time performance.

# 

# \---

# 

# \# ⚠️ Troubleshooting

# 

# \## 1. CUDAExecutionProvider not available

# 

# If you see an error related to:

# 

# ```text

# CUDAExecutionProvider

# ```

# 

# make sure your NVIDIA drivers, CUDA environment, and ONNX Runtime GPU installation are correctly configured.

# 

# For CPU-only execution, change the provider to:

# 

# ```python

# providers=\['CPUExecutionProvider']

# ```

# 

# \---

# 

# \## 2. No faces detected

# 

# Check that:

# 

# \- The input image contains a visible face.

# \- The face is sufficiently large.

# \- Lighting is adequate.

# \- The `known\_faces` images contain clear faces.

# \- The detection region contains the person.

# 

# \---

# 

# \## 3. Everyone is detected as Unknown

# 

# Try checking:

# 

# ```python

# threshold=0.5

# ```

# 

# The recognition threshold may need adjustment depending on your environment and camera quality.

# 

# Also make sure your reference images contain clear, high-quality faces.

# 

# \---

# 

# \## 4. Video does not open

# 

# Verify your video path:

# 

# ```python

# rtsp\_url="/home/ubuntu/office\_videos/office\_leave\_1.mov"

# ```

# 

# You can test the video separately with OpenCV:

# 

# ```python

# cap = cv2.VideoCapture("/path/to/video.mov")

# 

# if not cap.isOpened():

# &#x20;   print("Unable to open video")

# ```

# 

# \---

# 

# \## 5. MySQL connection error

# 

# Verify:

# 

# \- MySQL server is running

# \- Database `fr` exists

# \- Username is correct

# \- Password is correct

# \- The MySQL user has permission to access the database

# 

# Example:

# 

# ```bash

# sudo systemctl status mysql

# ```

# 

# \---

# 

# \# 🔒 Security Considerations

# 

# This project processes biometric information and should be deployed carefully.

# 

# Recommended practices:

# 

# \- Do not commit real face images to a public repository.

# \- Do not commit database passwords.

# \- Add sensitive directories to `.gitignore`.

# \- Use environment variables for credentials.

# \- Restrict access to the Flask server.

# \- Use HTTPS when deploying outside a trusted network.

# \- Obtain appropriate consent and follow applicable privacy/biometric-data laws.

# \- Limit access to stored face images and embeddings.

# 

# Recommended `.gitignore`:

# 

# ```gitignore

# venv/

# \_\_pycache\_\_/

# \*.pyc

# 

# detected\_faces/

# known\_faces/

# 

# .env

# 

# \*.mov

# \*.mp4

# \*.avi

# 

# .idea/

# .vscode/

# ```

# 

# \---

# 

# \# 📌 Example Workflow

# 

# ```text

# &#x20;               Video / Camera

# &#x20;                      │

# &#x20;                      ▼

# &#x20;                OpenCV Capture

# &#x20;                      │

# &#x20;                      ▼

# &#x20;               Detection Region

# &#x20;                      │

# &#x20;                      ▼

# &#x20;               InsightFace

# &#x20;                      │

# &#x20;             ┌────────┴────────┐

# &#x20;             │                 │

# &#x20;             ▼                 ▼

# &#x20;       Face Embedding      Face Bounding Box

# &#x20;             │                 │

# &#x20;             └────────┬────────┘

# &#x20;                      ▼

# &#x20;                 Deep SORT

# &#x20;                      │

# &#x20;                      ▼

# &#x20;                  Track ID

# &#x20;                      │

# &#x20;                      ▼

# &#x20;             Face Recognition

# &#x20;                      │

# &#x20;             ┌────────┴────────┐

# &#x20;             │                 │

# &#x20;             ▼                 ▼

# &#x20;           Known            Unknown

# &#x20;             │                 │

# &#x20;             └────────┬────────┘

# &#x20;                      ▼

# &#x20;               Save Face Image

# &#x20;                      │

# &#x20;                      ▼

# &#x20;                 MySQL Record

# &#x20;                      │

# &#x20;                      ▼

# &#x20;                Flask Stream

# ```

# 

# \---

# 

# \# 🚀 Future Improvements

# 

# Possible improvements for future versions:

# 

# \- \[ ] Move configuration to `.env`

# \- \[ ] Add a proper configuration file

# \- \[ ] Add multiple camera support

# \- \[ ] Add RTSP camera management

# \- \[ ] Add authentication to the Flask dashboard

# \- \[ ] Add attendance dashboard

# \- \[ ] Add entry/exit direction detection

# \- \[ ] Add daily/monthly attendance reports

# \- \[ ] Add CSV/Excel report export

# \- \[ ] Store embeddings in a database

# \- \[ ] Improve face-track association

# \- \[ ] Add automatic database reconnection

# \- \[ ] Add logging instead of `print()`

# \- \[ ] Add Docker deployment

# \- \[ ] Add systemd service for automatic startup

# \- \[ ] Add HTTPS deployment

# \- \[ ] Add configurable recognition thresholds

# \- \[ ] Add multi-face reference images per person

# 

# \---

# 

# \# 📜 License

# 

# Add the license that matches your intended use of this project.

# 

# For example:

# 

# ```text

# MIT License

# ```

# 

# If this project is intended for commercial or biometric applications, review the applicable software licenses and legal/privacy requirements before selecting a license.

# 

# \---

# 

# \# 🤝 Contributing

# 

# Contributions, improvements, and bug fixes are welcome.

# 

# 1\. Fork the repository.

# 2\. Create a feature branch.

# 

# ```bash

# git checkout -b feature/my-feature

# ```

# 

# 3\. Make your changes.

# 4\. Commit your changes.

# 

# ```bash

# git commit -m "Add my feature"

# ```

# 

# 5\. Push the branch.

# 

# ```bash

# git push origin feature/my-feature

# ```

# 

# 6\. Open a Pull Request.

# 

# \---

# 

# \# 👨‍💻 Author

# 

# \*\*maaitry11-dy\*\*

# 

# GitHub: \[https://github.com/maaitry11-dy](https://github.com/maaitry11-dy)

# 

# \---

# 

# \## ⭐ Support

# 

# If you find this project useful, consider giving the repository a ⭐ on GitHub.

