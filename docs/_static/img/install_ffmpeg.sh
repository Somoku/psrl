mkdir -p ~/bin && cd ~/bin
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz
mv ffmpeg-*-amd64-static/ffmpeg ~/bin/ffmpeg
mv ffmpeg-*-amd64-static/ffprobe ~/bin/ffprobe
rm -rf ffmpeg-*-amd64-static*
export PATH="$HOME/bin:$PATH"   # 也加到 ~/.bashrc
ffmpeg -version

# Example: edit a video to loop 5 seconds at the end
# ffmpeg -i video.mp4 -vf "tpad=stop_mode=clone:stop_duration=5" -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart -an video.looped.mp4