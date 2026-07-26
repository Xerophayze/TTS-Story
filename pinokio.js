const path = require("path")

module.exports = {
  version: "8.0.0",
  title: "TTS-Story",
  description: "Multi-voice text-to-speech for stories and audiobooks, with local and cloud TTS engines plus optional LLM text processing.",
  icon: "icon.svg",
  menu: async (kernel, info) => {
    // Check installation status
    let installing = info.running("install.json")
    let installed = info.exists("venv") && info.exists(".setup_complete")
    
    if (installing) {
      return [{
        icon: "fa-solid fa-plug",
        text: "Installing...",
        href: "install.json",
        default: true
      }]
    } else if (installed) {
      let running = info.running("start.json")
      
      if (running) {
        let memory = info.local("start.json")
        if (memory && memory.url) {
          return [
            {
              icon: "fa-solid fa-rocket",
              text: "Open Web UI",
              href: memory.url,
              default: true
            },
            {
              icon: "fa-solid fa-terminal",
              text: "Terminal",
              href: "start.json"
            },
            {
              icon: "fa-solid fa-rotate",
              text: "Update",
              href: "update.json"
            }
          ]
        } else {
          return [
            {
              icon: "fa-solid fa-terminal",
              text: "Terminal",
              href: "start.json"
            },
            {
              icon: "fa-solid fa-rotate",
              text: "Update",
              href: "update.json"
            }
          ]
        }
      } else {
        return [
          {
            icon: "fa-solid fa-power-off",
            text: "Start",
            href: "start.json",
            default: true
          },
          {
            icon: "fa-solid fa-rotate",
            text: "Update",
            href: "update.json"
          },
          {
            icon: "fa-solid fa-plug",
            text: "Reinstall",
            href: "install.json"
          },
          {
            icon: "fa-solid fa-broom",
            text: "Factory Reset",
            href: "reset.json"
          }
        ]
      }
    } else {
      return [
        {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.json",
          default: true
        },
        {
          icon: "fa-solid fa-rotate",
          text: "Update",
          href: "update.json"
        }
      ]
    }
  }
}
