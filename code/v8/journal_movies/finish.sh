set -e
cd final
# Movie 1: new flight chapter + existing readout chapter
ffmpeg -v error -y -i Movie1_flight_chapter.mp4 -i m1_readouts.mp4 -filter_complex "[0:v]scale=out_range=tv:out_color_matrix=bt709,format=yuv420p,setsar=1[a];[1:v]scale=in_range=tv:out_range=tv:out_color_matrix=bt709,format=yuv420p,setsar=1[b];[a][b]concat=n=2:v=1:a=0[o]" -map "[o]" -an -c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p -color_range tv -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags +faststart Supplementary_Movie_1_lesion_size_and_location.mp4
cp Movie3_paper.mp4 Supplementary_Movie_3_cell_selection.mp4
cp Movie4_paper.mp4 Supplementary_Movie_4_replacement_identity_and_wiring.mp4
for f in Supplementary_Movie_*.mp4; do echo "$f $(ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -show_entries format=duration -of csv=p=0 $f | tr '\n' ' ')"; ffmpeg -v error -i $f -f null - && echo decode-ok; done
# contact sheets
for f in Supplementary_Movie_*.mp4; do d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $f); ffmpeg -v error -y -i $f -vf "fps=6/$d,scale=640:-1,tile=3x2" -frames:v 1 ${f%.mp4}_contact_sheet.png; ffmpeg -v error -y -ss $(python3 -c "print(min(6,$d-0.5))") -i $f -frames:v 1 ${f%.mp4}_preview.png; done
ls -la
