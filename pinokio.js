const path = require("path")

module.exports = {
  version: "2.0",
  title: "TTS-Story",
  description: "Multi-Voice Text-to-Speech for Stories and Audiobooks. Supports Kokoro and Chatterbox TTS engines with GPU acceleration.",
  icon: "icon.svg",
  menu: async (kernel, info) => {
    // Check installation status
    let installing = info.running("install.json")
    let installed = info.exists("env")
    
    if (installing) {
      return [{
        icon: "fa-solid fa-plug",
        text: "Installing...",
        href: "install.json"
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
              href: memory.url
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
            href: "start.json"
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
          href: "install.json"
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
