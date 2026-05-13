#!/bin/bash
#
# Resilient How2Sign dataset downloader with retry and resume support.
# Usage: ./download_how2sign.sh <modality1> <modality2> ...
#
# Available modalities:
#   rgb_front_videos
#   rgb_side_videos
#   rgb_front_clips
#   rgb_side_clips
#   rgb_front_2D_keypoints
#   english_translation
#   english_translation_re-aligned
#
# Example:
#   ./download_how2sign.sh rgb_front_clips rgb_front_2D_keypoints english_translation_re-aligned
#
################################################################################

if (( $# < 1 )); then
    echo "USAGE: $0 <modality1> <modality2> ..."
    exit 1
fi

RETRIES=5          # Number of retry attempts per file
RETRY_WAIT=10      # Seconds to wait between retries

echo "Downloading the How2Sign dataset"

################################################################################
# Core download helper: resumable, retryable, skip-if-complete
################################################################################
download_file() {
    local FILE_ID="$1"
    local OUT_FILE="$2"

    # Skip if file already exists and is non-empty
    if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
        echo "  ✓ Already exists, skipping: $OUT_FILE"
        return 0
    fi

    for attempt in $(seq 1 $RETRIES); do
        echo "  ⬇ Downloading: $OUT_FILE (attempt $attempt/$RETRIES)..."

        # Fetch confirmation token first
        CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt \
            --keep-session-cookies --no-check-certificate \
            "https://docs.google.com/uc?export=download&id=${FILE_ID}" \
            -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

        wget -c --load-cookies /tmp/cookies.txt \
            "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
            -O "$OUT_FILE"

        local STATUS=$?
        rm -rf /tmp/cookies.txt

        if [ $STATUS -eq 0 ] && [ -s "$OUT_FILE" ]; then
            echo "  ✓ Done: $OUT_FILE"
            return 0
        fi

        echo "  ✗ Attempt $attempt failed for $OUT_FILE."
        if [ $attempt -lt $RETRIES ]; then
            echo "  ⏳ Waiting ${RETRY_WAIT}s before retry..."
            sleep $RETRY_WAIT
        fi
    done

    echo "  ✗ ERROR: Failed to download $OUT_FILE after $RETRIES attempts. Aborting."
    exit 1
}

################################################################################
# Unzip/extract helpers
################################################################################
unzip_file() {
    local ZIP="$1"
    local DEST="$2"
    echo "  📦 Extracting $ZIP → $DEST"
    unzip -o "$ZIP" -d "$DEST" && rm -f "$ZIP"
}

untar_file() {
    local TAR="$1"
    local DEST="$2"
    echo "  📦 Extracting $TAR → $DEST"
    tar -xf "$TAR" -C "$DEST" && rm -f "$TAR"
}

################################################################################
# Green Screen RGB videos - Frontal View (~290G train)
################################################################################
rgb_front_videos() {
    mkdir -p "./How2Sign/video_level/train/rgb_front"
    mkdir -p "./How2Sign/video_level/val/rgb_front"
    mkdir -p "./How2Sign/video_level/test/rgb_front"

    echo "***** Downloading Green Screen RGB videos (Frontal View) *****"

    download_file "1xWlMM2O3Gbp_8LK5FefoH0TVEmae6jIf" "train_raw_videos.z01"
    download_file "1krtYdpK_LQFgEUCnHxoYAW7EyhLMLWq0" "train_raw_videos.z02"
    download_file "1fXpWRNFhpuVm3ym7lT9vF_bnDjHkvP_K" "train_raw_videos.z03"
    download_file "1IFetFt4AzsxNCMZ0VVpX7YRgFAm58X48" "train_raw_videos.z04"
    download_file "1ZHuuun6Ae-AOLBns3LmuH7w8C9YCB4gH" "train_raw_videos.z05"
    download_file "1FQQIPblk-oLH_vu7h2tDO0oJaZ3xkp5N" "train_raw_videos.z06"
    download_file "19XNgERcolGAMPPgX-Gx_GebSTx3W4o0r" "train_raw_videos.z07"
    download_file "1YN-SA9uzrogEdKeT6UdQUIcuGEyYJILg" "train_raw_videos.z08"
    download_file "1SZQ2GzPLCkRqvsImAjULAPBiuAKi9DE9" "train_raw_videos.z09"
    download_file "1Xe1T5okJiopMXUiH3sc0mdCWNDYSBopd" "train_raw_videos.zip"
    download_file "1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g" "val_raw_videos.zip"
    download_file "1z0i6BBGHQ12ChY63hZH56QnczvQ0JfTb" "test_raw_videos.zip"

    echo "  📦 Merging train parts..."
    cat train_raw_videos.z* > train_raw_videos_all.zip
    unzip_file "train_raw_videos_all.zip" "./How2Sign/video_level/train/rgb_front"
    unzip_file "val_raw_videos.zip"       "./How2Sign/video_level/val/rgb_front"
    unzip_file "test_raw_videos.zip"      "./How2Sign/video_level/test/rgb_front"
    rm -f train_raw_videos.z*
}

################################################################################
# Green Screen RGB videos - Side View (~290G train)
################################################################################
rgb_side_videos() {
    mkdir -p "./How2Sign/video_level/train/rgb_side"
    mkdir -p "./How2Sign/video_level/val/rgb_side"
    mkdir -p "./How2Sign/video_level/test/rgb_side"

    echo "***** Downloading Green Screen RGB videos (Side View) *****"

    download_file "1Rmf6LfNWn6lWkAz6Iuj5pMOI2I5p4j1U" "train_side_raw_videos.z01"
    download_file "1FytIYIRYrBgAeNWIAhO5vnI2mYOvYC9i" "train_side_raw_videos.z02"
    download_file "1kC24jgNgjYYiIYhCRE-gGR28H_2xBBbP" "train_side_raw_videos.z03"
    download_file "1JunkM-ImFYao_MwDW9zeqe-6Th6rOLhR" "train_side_raw_videos.z04"
    download_file "1-vMckelz9fy4GVNYXRCcy7cJ12X4P3KZ" "train_side_raw_videos.z05"
    download_file "1uV413eKsihkNzquN2bwtIQG-OZZMz6sh" "train_side_raw_videos.z06"
    download_file "1sU8xrneFJHBzT_PFz4iRPqI8A7HGilhW" "train_side_raw_videos.z07"
    download_file "1RPLxeZ54uSZUJSXdPFhXOgeIXziOwTW9" "train_side_raw_videos.z08"
    download_file "1tClhr98PszBvFpo9ELKuhbTZZgTGGQqh" "train_side_raw_videos.z09"
    download_file "10xrXWgH7iW3E6sgJZDPRwlIhIaDLfHQm" "train_side_raw_videos.zip"
    download_file "1Z2H96JT68o7eTChEXPI9z3xyx7zUJPl5" "val_rgb_side_raw_videos.zip"
    download_file "1tCQ8KIuuiirXHsh29w0XAMNB3HLIGqgA" "test_rgb_side_raw_videos.zip"

    echo "  📦 Merging train parts..."
    cat train_side_raw_videos.z* > train_side_raw_videos_all.zip
    unzip_file "train_side_raw_videos_all.zip" "./How2Sign/video_level/train/rgb_side"
    unzip_file "val_rgb_side_raw_videos.zip"   "./How2Sign/video_level/val/rgb_side"
    unzip_file "test_rgb_side_raw_videos.zip"  "./How2Sign/video_level/test/rgb_side"
    rm -f train_side_raw_videos.z*
}

################################################################################
# Green Screen RGB clips - Frontal View (~31G train)
################################################################################
rgb_front_clips() {
    mkdir -p "./How2Sign/sentence_level/train/rgb_front"
    mkdir -p "./How2Sign/sentence_level/val/rgb_front"
    mkdir -p "./How2Sign/sentence_level/test/rgb_front"

    echo "***** Downloading Green Screen RGB clips (Frontal View) *****"

    download_file "1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW" "train_rgb_front_clips.zip"
    download_file "1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO" "val_rgb_front_clips.zip"
    download_file "1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG" "test_rgb_front_clips.zip"

    unzip_file "train_rgb_front_clips.zip" "./How2Sign/sentence_level/train/rgb_front"
    unzip_file "val_rgb_front_clips.zip"   "./How2Sign/sentence_level/val/rgb_front"
    unzip_file "test_rgb_front_clips.zip"  "./How2Sign/sentence_level/test/rgb_front"
}

################################################################################
# Green Screen RGB clips - Side View (~22G train)
################################################################################
rgb_side_clips() {
    mkdir -p "./How2Sign/sentence_level/train/rgb_side"
    mkdir -p "./How2Sign/sentence_level/val/rgb_side"
    mkdir -p "./How2Sign/sentence_level/test/rgb_side"

    echo "***** Downloading Green Screen RGB clips (Side View) *****"

    download_file "1oiw861NGp4CKKFO3iuHGSCgTyQ-DXHW7" "train_rgb_side_clips.zip"
    download_file "1mxL7kJPNUzJ6zoaqJyxF1Krnjo4F-eQG" "val_rgb_side_clips.zip"
    download_file "1j9v9P7UdMJ0_FVWg8H95cqx4DMSsrdbH" "test_rgb_side_clips.zip"

    unzip_file "train_rgb_side_clips.zip" "./How2Sign/sentence_level/train/rgb_side"
    unzip_file "val_rgb_side_clips.zip"   "./How2Sign/sentence_level/val/rgb_side"
    unzip_file "test_rgb_side_clips.zip"  "./How2Sign/sentence_level/test/rgb_side"
}

################################################################################
# B-F-H 2D Keypoints clips - Frontal View (~21G train)
################################################################################
rgb_front_2D_keypoints() {
    mkdir -p "./How2Sign/sentence_level/train/rgb_front/features"
    mkdir -p "./How2Sign/sentence_level/val/rgb_front/features"
    mkdir -p "./How2Sign/sentence_level/test/rgb_front/features"

    echo "***** Downloading B-F-H 2D Keypoints clips (Frontal View) *****"

    download_file "1TBX7hLraMiiLucknM1mhblNVomO9-Y0r" "train_2D_keypoints.tar.gz"
    download_file "1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w" "val_2D_keypoints.tar.gz"
    download_file "1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp"  "test_2D_keypoints.tar.gz"

    untar_file "train_2D_keypoints.tar.gz" "./How2Sign/sentence_level/train/rgb_front/features"
    untar_file "val_2D_keypoints.tar.gz"   "./How2Sign/sentence_level/val/rgb_front/features"
    untar_file "test_2D_keypoints.tar.gz"  "./How2Sign/sentence_level/test/rgb_front/features"
}

################################################################################
# English Translation (original alignment)
################################################################################
english_translation() {
    mkdir -p "./How2Sign/sentence_level/train/text/en/raw_text"
    mkdir -p "./How2Sign/sentence_level/val/text/en/raw_text"
    mkdir -p "./How2Sign/sentence_level/test/text/en/raw_text"

    echo "***** Downloading English Translation (original) *****"

    download_file "1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f" "how2sign_train.csv"
    download_file "1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3" "how2sign_val.csv"
    download_file "1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG" "how2sign_test.csv"

    mv how2sign_train.csv "./How2Sign/sentence_level/train/text/en/raw_text/"
    mv how2sign_val.csv   "./How2Sign/sentence_level/val/text/en/raw_text/"
    mv how2sign_test.csv  "./How2Sign/sentence_level/test/text/en/raw_text/"
}

################################################################################
# English Translation (manually re-aligned) — RECOMMENDED
################################################################################
english_translation_re-aligned() {
    mkdir -p "./How2Sign/sentence_level/train/text/en/raw_text/re_aligned"
    mkdir -p "./How2Sign/sentence_level/val/text/en/raw_text/re_aligned"
    mkdir -p "./How2Sign/sentence_level/test/text/en/raw_text/re_aligned"

    echo "***** Downloading English Translation (re-aligned) *****"

    download_file "1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok" "how2sign_realigned_train.csv"
    download_file "1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI" "how2sign_realigned_val.csv"
    download_file "1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu" "how2sign_realigned_test.csv"

    mv how2sign_realigned_train.csv "./How2Sign/sentence_level/train/text/en/raw_text/re_aligned/"
    mv how2sign_realigned_val.csv   "./How2Sign/sentence_level/val/text/en/raw_text/re_aligned/"
    mv how2sign_realigned_test.csv  "./How2Sign/sentence_level/test/text/en/raw_text/re_aligned/"
}

################################################################################
# Dispatch
################################################################################
for ARG in "$@"; do
    shift
    case "${ARG}" in
        "rgb_front_videos")              rgb_front_videos;;
        "rgb_side_videos")               rgb_side_videos;;
        "rgb_front_clips")               rgb_front_clips;;
        "rgb_side_clips")                rgb_side_clips;;
        "rgb_front_2D_keypoints")        rgb_front_2D_keypoints;;
        "english_translation")           english_translation;;
        "english_translation_re-aligned") english_translation_re-aligned;;
        *) echo "ERROR: Unknown modality '${ARG}'. Check the header of this script for valid options.";;
    esac
done

echo ""
echo "✅ Download complete! Thank you for using the How2Sign dataset."
################################################################################
