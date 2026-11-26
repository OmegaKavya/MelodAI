// What the server accepts (ALLOWED and FORMATS in server/main.py), kept out of
// the component so a plain node test can hold the two in step.
//
// Extensions only. A MIME type is broader than the extension it is named for:
// audio/ogg also offers .opus and .oga, and audio/mpeg offers .mp2, all of
// which the server refuses on arrival.
export const ACCEPT = ".mp3,.wav,.flac,.ogg";

export const UPLOAD_PROMPT = "Drop an MP3, WAV, FLAC or OGG, or choose a file.";
