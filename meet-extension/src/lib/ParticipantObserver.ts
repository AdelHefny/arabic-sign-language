export const observeParticipants = (onNewVideo: (el: HTMLVideoElement) => void) => {
  const observer = new MutationObserver((mutations) => {
    const videos = document.querySelectorAll('video');
    videos.forEach((video) => {
      // Ensure we haven't already attached to this video
      if (!video.dataset.slAnnotated) {
        video.dataset.slAnnotated = "true";
        onNewVideo(video);
      }
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
};
